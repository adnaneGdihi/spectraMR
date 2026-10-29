# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.3.3] - 2026-10-28

### Added
- **One device mesh for every collective:** `parallel.mesh {tensor, context, spatial, pipeline}`
  (each default 1). Only `ddp` and `fsdp` may split an axis, and the world size must divide
  exactly by the mesh product.
- **Tensor parallelism** for the transformer blocks under `ddp` (`parallel.mesh.tensor > 1`):
  every `TokenAttention` and the MLPs of `DiTBlock` / `UViTBlock` are split Megatron-style. Each
  split block is checked against an unsplit copy before training. A model with nothing to split,
  a head or hidden count that does not divide, and `fsdp` with a tensor mesh are refused.
- **Spatial parallelism for training** (`parallel.mesh.spatial: N`, `ddp` only): a 3D model trains
  with each volume split along depth across the spatial group, after an equivalence check on a
  synthetic volume. Its combinations with context, tensor or pipeline parallelism,
  `sync_batch_norm` and `optimization.compile` are refused.
- The hardware-warp calibration honours `split.type: loso_subject` over its pose set, and the
  fitted warp is the arm's model (`hardware_warp_operator`).

### Fixed
- `validation.enabled: false` is read: no validation loader is attached, so no validation event
  or best checkpoint follows and the validation reachability gate does not fire.
- Test collection no longer re-reads each test class's source once per collected item; the full
  suite collects in about a minute instead of 37-55 minutes.
- The release test lane: a `loguru` import (not a dependency) is gone, the mutation job runs
  under mutmut 2 against the physics tests, and the lock re-resolves (environments capped below
  Python 3.15, where gudhi and ray[tune] ship no wheels).

### Security
- The lock moves to urllib3 2.8.0, pillow 13.0.0 and filelock 4.2.4.

## [1.3.2] - 2026-10-28

### Added
- **FSDP2.** Sharded data parallelism now uses `fully_shard` on a `DeviceMesh`, and fp16
  runs under FSDP2 with a sharded gradient scaler. Complex-valued gradients are unscaled
  correctly.
- **Spatial (depth-split) parallelism for 3D conv models.** A `[B, C, D, H, W]` volume is
  split along depth across a spatial group, and neighbouring ranks exchange halo planes. Use
  it when one volume's activations do not fit on a GPU.
- **Activation offload under every strategy:** `optimization.gradient.activation_offload: none | cpu`
  (default `none`; an unknown value is refused).
- **`spectramr campaign profile <campaign.yaml>`** profiles every enabled arm of a campaign as
  one SLURM array and files each profile under `<campaign>/<arm>/profile/<run_id>/`.
- A multi-GPU, multi-node smoke harness for the parallel paths.
- Validation logs its memory use, and keeps fewer per-batch tensors alive.

### Changed
- **FSDP1 is removed.** Its config keys (for example `parallel.fsdp.use_orig_params`) are
  refused at load with the reason. Configs that declare them must drop them.
- **`optimization.gradient.enable_checkpointing` raises at build** on a model without
  `set_grad_checkpointing`. Previously it fell back to wrapping single layers and turned a
  wrapping failure into a warning.
- DeepSpeed GAN runs admit several engines only when the config enables it, and freeze the
  critic in one place.

### Fixed
- fp32 DeepSpeed raises on non-finite gradients at the optimizer boundary instead of
  stepping NaNs into every weight.
- `offload_optimizer` under DeepSpeed builds `DeepSpeedCPUAdam`.
- `UNet3DGenerator` has a full-resolution stem, and derives its per-level skips from
  `features`.
- The non-Cartesian data-consistency step scale is deterministic and shards under FSDP2.
- 36 swept metrics that declared no ROI policy now declare one.

## [1.3.1] - 2026-10-28

### Removed
- **The self-hosted `thor` runner.** `release.yml` and `test-release.yml` run every job
  on `ubuntu-latest`; their `runner` dispatch input and the `RELEASE_RUNNER` repository
  variable are gone. The public repository's desired settings declare no self-hosted
  runner, so any registered one reports as drift, and
  `tests/unit/ci/test_no_self_hosted_runner.py` fails on any job that targets one.

## [1.3.0] - 2026-10-28

### Added
- `spectramr.infrastructure.checkpointing`: the checkpoint layout (`step_N`/`best` in any of
  `safetensors`, `pt`, `pth`), a versioned `RunState` payload, codecs, one selector,
  `keep_last_n` retention and the `weights()` loader. Not yet wired into training.
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

- **Degradations in 2-D, multislice and 3-D.** Every registry axis declares an implementation
  per `AcquisitionMode` (`slice_2d`, `multislice_2d`, `volume_3d`) and runs on `(B, C, H, W)`
  or `(B, C, D, H, W)` with an `AcquisitionGeometry`; a mode the physics excludes raises
  `UnsupportedAcquisitionMode`, an unmet `requires` raises `MissingInput`. The catalogue adds
  axes D33-D70 (motion, metal and chemical shift, flow, hardware, receive chain, transmit,
  sequence, multislice, reconstruction and field-dependent contrast), appended to the registry
  order so existing sweep indices do not move.
- `physics.digital_twin.acquisition`: the scanner the twin's registry stage simulates (preset,
  mode, voxel size, slice and line order, readout bandwidth, sequence). It builds one
  `ScannerProfile` at `physics.field_strength`; a geometry knob in a mode that reads none
  raises at load.
- `sim2rank --mode 3d --acquisition {multislice_2d,volume_3d}` sweeps every registry axis on a
  volume, with `--scanner-profile` and `--volume-size H W D`; it records the axes the mode
  cannot run and how each metric was aggregated over the volume. The meta-evaluation stays
  2-D and raises on `3d`.

### Changed
- Degradations: `ScannerProfile` conditions the registry through `apply_degradation(...,
  profile=)` and each spec's `condition`; `axis_kwargs` is gone. With a profile, a 3 T
  sweep on `b0_geometric_distortion` (dwell from the readout bandwidth), `chemical_shift`
  (physical fat shift and the profile's fat fraction) and `concomitant_phase` (the
  profile's B0) differs from a profile-less one; only `b0` reproduces the default ramp at 3 T.
- Checkpoints: every save, resume, best restore and weight load goes through
  `spectramr.infrastructure.checkpointing`. Files are `checkpoints/step_NNNNNNNN.<format>`
  and `checkpoints/best.<format>`, where `checkpoint.format` (`safetensors` default, `pt`,
  `pth`) is now honoured. The final checkpoint records the last iteration that
  ran and is written before the best weights are restored; restore-best restores
  weights and EMA only. `checkpoint.load_optimizer_state` is honoured.
- Campaign `checkpoint_from` resolves the parent's best (else latest) checkpoint when
  the child starts.
- DeepSpeed: the RunState file carries gathered weights at every ZeRO stage; ZeRO tag
  directories are `checkpoints/deepspeed/<model>/step_NNNNNNNN` and `.../best`.
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

- **Degradation sweeps are not comparable across this release.** The EMI axis (D32) joins the
  registry, k-space grids are centred, noise and random draws are per sample (a batch neighbour
  no longer changes a sample), and each sweep axis draws its seed through `derive_axis_seed`.
  `b0_drift` now displaces content by `-df / BW_px` (and `-df N ESP` in EPI), the sign of every
  other off-resonance axis. `b1_transmit` and `dielectric_b1` reference their seeded B1+ map
  to the transmit calibration (mean flip angle over the object is nominal), so the standing
  wave no longer darkens low-field images globally. `tse_t2_blur` with an echo train longer
  than the phase-encode lines acquires it single-shot.
- `DigitalTwinDegradation(pe_axis=)` is required: the TorchIO axis carrying phase-encode is
  declared, never inferred.
- Import paths: the registry queries (`list_degradations`, `slice_capable_degradations`,
  `requirement_free_degradations`, `affected_components`, `is_phase_only`, ...) moved from
  `degradations.registry` to `degradations.registry_queries`, and the scanner presets
  (`FASTMRI_3T`, `M4RAW_03T`, `PRESETS`, `resolve_profile`) from `scanner_profile` to
  `scanner_presets`. The package `spectramr.infrastructure.physics.degradations` still
  exports the queries.
- `physics.digital_twin.acquisition.voxel_mm` is accepted in `slice_2d`, where the
  mm-declared 2-D kernels read it.

### Removed
- `spectramr.infrastructure.physics.digital_twin_extensions`: its contents live in
  `spectramr.infrastructure.physics.degradations`; import from there.
- `ScannerProfile.from_acquisition_params`, which had no production caller and set B0 from
  a header beside the `physics.field_strength` owner.
- `CheckpointDirector`, `CheckpointService`, `ICheckpointManager`, the checkpoint
  delegate, the `best.pt` alias, the best-EMA sidecar, the DeepSpeed consolidated best
  file and the checkpoint key ladder (`CHECKPOINT_STATE_KEYS`/`resolve_state_dict`).
- Config keys: `checkpoint.pretrained_path`, `checkpoint.resume_training`,
  `checkpoint.resume_from` (now `training.distillation.teacher_checkpoint`),
  `parallel.deepspeed.save_consolidated_best`, and six existence-only prior checkpoint
  fields.
- `CheckpointDirector.cleanup_old_checkpoints`, which no code path called, and
  `CheckpointService`'s retention sweep, which globbed a format no training-path
  checkpoint uses and was unscoped to the run. `checkpointing.retention` is now the
  one owner of retention.
- **The exception-selected checkpoint fallback**. When the periodic
  `CheckpointDirector` save raised, the loop wrote a generator-only
  `checkpoint_step_N.safetensors` into `checkpoint.checkpoint_dir` through
  `CheckpointService` and trained on. That file had no discriminator, schedulers or
  EMA, could land in another directory than every sibling checkpoint, and, at a
  higher step than the last director file, was what `--resume auto` selected. A
  failed periodic or final save now raises and the run fails with the cause;
  `<training.output_dir>/checkpoints` is the only destination.
- **`checkpoint.keep_best_n`**. Nothing read it: the best checkpoint is the
  single file `best.<format>`, replaced on each improvement. A declaration is
  now refused at load by name (`config/schemas/removals.py`), and it is drained from
  every loadable `1.0` config in `experiments/`, the 1.0 template and the in-repo
  reference configs. A saved `resolved_config.json` from an earlier run still
  rebuilds: the snapshot reload drops removed keys with a warning.
- **`checkpoint.checkpoint_dir` and its `save_dir` alias**. No checkpoint
  was written there, and resume and `run_summary.json` look in
  `<training.output_dir>/checkpoints`. The key's one effect was that
  the pre-1.0 checkpoint service created it at startup, so every arm on the
  `./checkpoints` default left an empty `checkpoints/` under the launch directory. The
  checkpoint store creates nothing at construction, and a config declaring either spelling is
  refused with the reason, and drained from every loadable `1.0` config in
  `experiments/`. Checkpoint locations of existing runs are unchanged.
  `cli/manifest_dispatch` and HPO trials no longer override the key.

### Fixed
- `DigitalTwinDegradation` (`physics.digital_twin.apply_as_transform`): the transform builder
  hard-coded `pe_axis="W"`. It now reads the dataset's `data.acquisition.phase_encode_axis`
  (`-2` -> `W`, `-1` -> `H`), and stamps `pe_axis_source` in the transform's applied record;
  without a `data.acquisition` block it keeps `W`, stamped `undeclared`. **Outputs change** for
  any twin-transform arm on M4Raw T1/T2/FLAIR or fastMRI brain (`phase_encode_axis: -1`): the
  PE/RO-asymmetric twin operators (spin-warp B0 shift, motion ghosts, Nyquist ghost, partial
  Fourier, `tse_t2_blur`, `fov_foldover`) now act on the declared axes. No arm in `experiments/`
  sets `apply_as_transform: true` today. A `volume_3d` scanner profile at this boundary now
  raises: the raw k-space it reads is multislice.
- `simulate_ulf_from_hf`: `SimulateULFFromHF` displaced its B0 and Maxwell shifts along the
  phase-encode axis (EPI physics), but spin-warp/TSE off-resonance displaces along readout. A new
  `sequence: Literal["spin_warp", "epi"]` knob (default `"spin_warp"`) selects the axis: with
  `pe_axis="H"` the shifts now move along W; `sequence: epi` restores the previous phase-encode
  behaviour. The warp is now identity-exact (`align_corners=True`, (x, y, z) grid). A missing source
  image or a non-4-D tensor now raises instead of logging a warning and skipping, and an unknown
  `pe_axis` raises. **Outputs change** for
  every arm that uses `SimulateULFFromHF` with a phase-encode shift.
- `realistic_degradations`: `B0GeometricDistortion` and `ElasticDeformation3D` stacked their
  `grid_sample` grid as (z, y, x) although 5-D `grid_sample` reads (x, y, z), which transposed
  non-cubic volumes even at zero field, and used `linspace(-1, 1)` with `align_corners=False`
  (a half-voxel resample). Grids are now (x, y, z) with `align_corners=True`, so a zero field is the
  exact identity. `B0GeometricDistortion(readout_dim=)` is replaced by a required `shift_dim` in
  `{3, 4}` (H or W); `RandomB0Distortion` passes `4`, the axis it effectively shifted before.
  **Outputs change** for every config with `augmentation.enable_b0_distortion: true`.
- **`predict`/`infer` rebuild runs whose snapshot names a removed key.** Settings are
  rebuilt from the `_declared` block of a run's `resolved_config.json`, and the
  schema refused any key retired after the run was written. Since
  `early_stopping.check_interval` was removed, that refused every run trained
  from the 16 n2n/self_supervised arms that declared it. The rebuild now drops keys
  listed in `removals.py`, with one warning naming each. Nothing read them, so the
  rebuilt run is the one that trained. Authored YAMLs are still refused.
- **The final checkpoint carries the strategy-owned state and the live AMP
  scaler**. Its pre-1.0 save path was the only one that omitted the strategy,
  so the file `--resume auto` selects when a finished run is extended held no
  strategy state and the scaler of an object that never saw a gradient. Every save
  (periodic, best, loss-schedule and final) now goes through `checkpointing.capture`,
  so their contents cannot drift apart again.
- **`checkpoint.keep_last_n` is applied on the training path.** Every periodic
  checkpoint used to stay on disk. After each step save the run keeps the
  `keep_last_n` highest-step `step_NNNNNNNN.<format>` files in its `checkpoints/`
  and deletes the older ones. The best file, other names, and steps above the
  triggering save are never touched. Under DeepSpeed each
  pruned file's ZeRO tag directory is deleted with it, so a resume always finds a
  matched pair; the `best` tag and the tag DeepSpeed's `latest` file names are kept.
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
  published tree, so a bump that lands only there is reverted by the next publish.
  `pydantic-settings>=2.14` in `pyproject.toml` already admits 2.14.2, and
  the other two are transitive, so no declared constraint changes.

### Migration
- Convert pre-1.0 runs once with `convert_checkpoints_v1.py <run_dir>` (maintainer tree, `scripts/migrations/`; not in the public export)
  before `--resume` or `predict` on them.
- Convert a run only while no job can write to it: finish or `scancel` its chain first.
  A conversion killed at any point finishes when the same command is rerun; one that
  another writer touched refuses, changes nothing, and prints an inventory.

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
[Unreleased]: https://github.com/adnaneGdihi/spectramr/compare/v1.3.3...HEAD
[1.3.3]: https://github.com/adnaneGdihi/spectramr/releases/tag/v1.3.3
[1.3.2]: https://github.com/adnaneGdihi/spectramr/releases/tag/v1.3.2
[1.3.1]: https://github.com/adnaneGdihi/spectramr/releases/tag/v1.3.1
[1.3.0]: https://github.com/adnaneGdihi/spectramr/releases/tag/v1.3.0
[0.1.2]: https://github.com/adnaneGdihi/spectramr/releases/tag/v0.1.2
[0.1.1]: https://github.com/adnaneGdihi/spectramr/releases/tag/v0.1.1
[0.1.0]: https://github.com/adnaneGdihi/spectramr/releases/tag/v0.1.0
