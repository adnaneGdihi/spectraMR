# Write an experiment YAML

Every experiment in spectraMR is one YAML file. The schema is a set of frozen
Pydantic v2 models defined in `src/spectramr/config/schemas/`, and every file
declares `config_version: '1.0'`. The audit ladder catches mistakes before
training starts.

This page is the short, opinionated walkthrough. The dry exhaustive
reference is at [configuration schema reference](../config_schema_reference.rst).

## Where the file lives

| Where | When |
|---|---|
| `experiments/inprogress/<paradigm>/<arm>.yaml` | **Default** for any new experiment under development. The smoke wrapper auto-discovers everything here. |
| `experiments/inprogress/<paradigm>/ablations/<arm>.yaml` | Ablation studies sit beside their parent. |
| `experiments/active/<arm>.yaml` | After smoke + audit pass. Don't put new files here directly. |
| `experiments/campaigns/<cohort>.yaml` | A campaign manifest — see the campaigns user guide. |
| `experiments/templates/` | Copy-paste starting points. |

The promotion path is **always** `inprogress → active`, never the other
direction. See the [four-step paradigm recipe](https://github.com/adnaneGdihi/spectramr/blob/main/CONTRIBUTING.md#adding-a-new-training-paradigm-four-step-recipe).

## The house-style skeleton

**This is a skeleton, not a runnable file.** Every `<...>` below is a
placeholder you must replace — `model_type: <registered_name>` fails the audit
as written, which is the point: the audit rejects an unregistered name rather
than substituting a default.

Only four of these blocks are required by the schema itself (`model`, `data`,
`optimization`, `logging`). The rest are optional to *load* and expected in
practice — a paradigm that needs one still fails Tier 1 without it, and `audit`
is `--strict` by default, so a missing `validation:` or `physics:` block
usually surfaces as a warning that exits 2 rather than as a schema error.
Declare them all unless you have a reason not to.

```yaml
config_version: '1.0'

metadata:
  name: <arm_name>
  description: |
    What this experiment does, why, and what it's compared against.
  tags: {paradigm: ..., type: ..., novelty: ...}
  version: '1.0'

undersampling:
  base_acceleration: 4
  center_fraction: 0.08

artifacts:
  persistent_root: experiments/results/<arm_name>

checkpoint:
  enabled: true
  keep_best_n: 3
  keep_last_n: 5
  save_interval: 10000

data:
  dataset_type: nifti_paired         # see schema for full list

  loader:
    batch_size: 2
    num_workers: 4
  coils:
    processing_mode: rss          # or 'sense', 'as_is'
  sampling:
    patch_size: [256, 256, 1]
  source:
    root: ${SPECTRAMR_DATA_ROOT}/processed/ulf_to_hf_ldm/train
losses:
  image_losses:
    - {name: l1,  weight: 1.0, enabled: true}
    - {name: ssim, weight: 0.5, enabled: true}
  kspace_losses: []
  complex_losses: []

  policy:
    output_domain: image               # or 'kspace', 'complex', 'latent'
logging:

  identity:
    experiment: <arm_name>
  sinks:
    level: info
loss_logging:
  enabled: true
  csv_path: experiments/results/<arm_name>/logs/losses.csv

metrics:
  best_metric_name: val_psnr
  best_metric_mode: max
  domain: image

model:
  model_type: <registered_name>
  in_channels: 1
  out_channels: 1
  input_type: image
  model_kwargs: {}

optimization:

  optimizer:
    type: adamw
    learning_rate: 1.0e-4
    weight_decay: 1.0e-5
  precision:
    enabled: false
training:
  training_mode: <registered_mode>
  strategy_class: spectramr.infrastructure.training.strategies.<...>
  epochs: 100
  device: cuda
  output_dir: experiments/results/<arm_name>

validation:
  enabled: true

  schedule:
    interval_steps: 5000
  scoring:
    compute: [psnr, ssim]
physics: {}                          # required block; may be empty
run:
  seed: 42
```

## The gotcha checklist

Every YAML must pass these — and they're the easiest places to slip up.

### 1. Diffusion paradigms need `training.diffusion` + `num_timesteps`

If `training.strategy_class` matches `*Diffusion*`, the audit rejects the
config unless you declare:

```yaml
training:
  diffusion:
    type: score_based            # or 'cold'
    degradation: null
  num_timesteps: 100
```

This applies even to strategies that don't subclass
`DiffusionTrainingStrategy` — the audit matches by name pattern (e.g.
`teichmuller_cold_diffusion`, `hrf_manifold_diffusion`).

### 2. Loss-domain block matching

`losses.image_losses[*]` may only contain losses with
`@register_loss(domain="image")`. Three escape hatches when a loss
genuinely needs to appear elsewhere:

- Move the loss to its matching block.
- Set `losses.output_domain: latent` (or whatever the loss domain is).
- Mark the loss `compatible_with=["image"]` in its decorator (see
  [add_loss](add_loss.md#domain-spillover-when-the-audit-fights-you)).

### 3. RSS channel collapse on image-domain datasets

`coil_processing_mode: rss` on an image-domain `dataset_type`
(`nifti`, `nifti_paired`, `npy_slice`) collapses input to 1 channel via
the RSS magnitude. If `model.in_channels=2` here, the `domain_alignment`
audit rejects it:

| dataset_type | coil_processing_mode | model.in_channels |
|---|---|---|
| `kspace` | any | `2` (real/imag interleave) |
| `nifti_paired`, `npy_slice` | `rss` | `1` (magnitude) |
| `nifti_paired`, `npy_slice` | `as_is` | as the file declares |

### 4. MRF strategies need an `mrf:` block

If `training_mode` is one of the MRF aliases, the audit demands specific
fields:

```yaml
mrf:
  spiral_rotation_schedule: golden_angle     # or 'linear', 'random'
  n_timepoints: 1000
  sar_max_w_per_kg: 4.0
  gradient_slew_rate_t_per_m_per_s: 200.0
  phantom_calibration_path: ${SPECTRAMR_DATA_ROOT}/calibration/phantom.h5
```

The required subset depends on which MRF audit-plan validators are wired
in for the paradigm — see the audit JSON for the exact list.

### 5. EPI distortion needs `data.phase_encode_axis`

For `beltrami_epi_distortion` and related strategies:

```yaml
data:
  phase_encode_axis: -2    # or -1 for transverse PE
```

### 6. Strategies that read extra batch keys need `data.expose_*` flags

The strategy layer reads `batch["conformal_jacobian"]` (and similar)
only if the data pipeline mounts the relevant wrapper. Flip the flag:

```yaml
data:
  expose:
    conformal_jacobian: true
    cortex_flatten_grid: true
    glm_design_matrix: true
    scanner_id: true
```

Without the flag, the strategy gets a `KeyError` mid-training. The
audit warns but doesn't reject — so this is the bug that bites at
epoch 1.

### 7. Strategy class + `training_mode` must agree

`training_mode` is matched against `TRAINING_MODE_CONSTRAINTS` in
`src/spectramr/config/validation_constants.py`. Adding a new strategy
means adding **three** entries: `STRATEGY_CLASS_PATHS`,
`VALID_TRAINING_MODES`, and `TRAINING_MODE_CONSTRAINTS`. See
[add_paradigm](add_paradigm.md#2-register-the-dispatch-key).

### 8. Use `${SPECTRAMR_DATA_ROOT}` for data paths

Never hardcode a cluster path. The audit's `hardcoded_cluster_paths`
check rejects any data field starting with `/project/<user>/` or
`/scratch/<user>/` *unless* that prefix matches your own
`SPECTRAMR_DATA_ROOT` / `PROJECT_ROOT` env var (see
[environment variables](../environment_variables.rst)).

## Validate before launching

```bash
spectramr audit experiments/inprogress/<paradigm>/<arm>.yaml             # Tier 0+1, ~100ms
spectramr audit experiments/inprogress/<paradigm>/<arm>.yaml --probe     # adds Tier 2 (~30s, GPU)
```

`--strict` (default in the smoke wrapper) exits non-zero on any warning.

## Common errors, decoded

| Error message | Cause | Fix |
|---|---|---|
| `extra="forbid"` | YAML has a key the schema doesn't know | Either drop the key, or add it to the appropriate sub-schema |
| `ValidationError: training_mode` | Typo in the alias | Check `VALID_TRAINING_MODES` |
| `paradigm_required_fields` | Diffusion paradigm missing `training.diffusion` block | Add it (gotcha 1) |
| `domain_alignment` | RSS + 2-channel mismatch | Fix `model.in_channels` (gotcha 3) |
| `loss_domain_block_match` | Loss in wrong block | Move it (gotcha 2) |
| `hardcoded_cluster_paths` | Literal `/project/...` in YAML | Use `${SPECTRAMR_DATA_ROOT}` (gotcha 8) |

## Reference YAMLs to copy from

| What you are building | Config to copy |
|---|---|
| Anything, from scratch | `experiments/templates/comprehensive_config_template.yaml` (`model_type: unet`, `training_mode: reconstruction`) |
| Self-supervised reconstruction | `experiments/inprogress/reconstruction/ssdu_selfsup_m4raw.yaml` (`model_type: unrolled_reconstruction`) |
| Score-based diffusion | `experiments/inprogress/diffusion/experiment_96_sde_diffusion.yaml` (`model_type: score_based_diffusion`) |
| Complex-valued reconstruction | `experiments/inprogress/workflow_baselines/b1_structural_recon_m4raw.yaml` (`model_type: complex_unet`) |

Copy-and-modify is cheaper than writing from scratch.

## Next steps

- [Configuration schema reference](../config_schema_reference.rst) — every field, every type.
- [Audit ladder](../audit_ladder_user_guide.rst) — what each tier checks.
- [Add a paradigm](add_paradigm.md) — when a new training mode is needed.
