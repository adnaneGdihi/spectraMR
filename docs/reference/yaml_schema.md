# YAML schema reference

The authoritative schema is the Pydantic v2 model tree under
`src/spectramr/config/schemas/`. This page is the human-readable summary
of the v1.0 layout — every top-level block, every required field, every
common enum. It is a *summary*: the exhaustive, generated list of every
dotted path lives in [config_key_reference](../config_key_reference.rst).

For the short tutorial-style intro see
[write_experiment_yaml](../how_to/write_experiment_yaml.md). For the
audit ladder that enforces this schema see
[audit_ladder](../explanation/audit_ladder.md).

## Top-level: `TrainingSettings`

A v1.0 YAML deserialises into a frozen `TrainingSettings` Pydantic
model. The settings object is **passed by reference** through every
layer — no module re-parses the YAML, no layer mutates it.

`config_version` is required by the loader and must be `'1.0'`
(`ACCEPTED_CONFIG_VERSIONS = frozenset({'1.0'})`); an older declaration raises.

**Only four top-level blocks are required by the schema:**

```
model                    # model_type + in/out channels + model_kwargs
data                     # loader / source / coils / sampling sub-blocks
optimization             # optimizer / gradient / precision sub-blocks
logging                  # identity / intervals / images sub-blocks
```

The other 26 top-level fields are optional, among them `training`,
`validation`, `metrics`, `losses`, `checkpoint`, `physics`, `undersampling`,
`adapters`, `run`, `workflow`, `metadata` and `reporting`. Optional does not
mean inert — a paradigm that needs a block still fails Tier 1 without it, and
the audit is `--strict` by default.

`TrainingSettings` itself is `extra='forbid'`, so an unrecognized top-level key
raises rather than being ignored. That is not true at every depth: 263 of 353
schema classes forbid extras, 23 allow them deliberately, and 67 leave the
setting unset, where Pydantic's default is to *ignore* silently.

## `metadata`

```yaml
metadata:
  name: my_arm                       # required, snake_case
  description: |
    Free-form text. Multi-line is fine.
  tags:
    paradigm: diffusion              # high-level paradigm family
    type: baseline                   # baseline | ablation | breakthrough | …
    novelty: my_arm                  # short slug identifying the novelty
  version: '1.0'                     # must match config_version
```

Every example on this page is an **excerpt** of one block. A runnable file also
needs `config_version: '1.0'` at the top plus the four required blocks listed
above; copy
`src/spectramr/config/schemas/templates/v1.0_reference.yaml` rather than
assembling one from these fragments.

`tags` is a free-form dict; the audit doesn't enforce specific keys but
the convention above keeps the smoke-mosaic plotter happy.

## `data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `dataset_type` | enum | yes | `kspace`, `nifti`, `nifti_paired`, `npy_slice`, `m4raw`, `preprocessed`, `dicom`, `pde_synthetic`, … |
| `source.root` | path | yes | Use `${SPECTRAMR_DATA_ROOT}/...`. Flat `data_root` still folds. |
| `sampling.patch_size` | `list[int, int, int]` | yes | (H, W, D); D=1 for 2D. Flat `patch_size` folds. |
| `loader.batch_size` | int | yes | per-GPU. Flat `batch_size` folds. |
| `coils.processing_mode` | enum | yes | `rss`, `sense`, `as_is`. Flat `coil_processing_mode` folds. |
| `loader.num_workers` | int | no | dataloader workers. Flat `num_workers` folds. |
| `split.validation_fraction` | float | no | 0.0–1.0. Flat `validation_split` folds. |
| `source.index_path` | path | no | pre-computed dataset index for fast startup |
| `image_undersampling` | bool | no | apply the declared mask to an image-domain source — without it a declared acceleration is never applied |
| `phase_encode_axis` | int | no | required for EPI distortion strategies |
| `expose_*` | bool | no | extra-batch-key flags (see [write_experiment_yaml § gotcha 6](../how_to/write_experiment_yaml.md#6-strategies-that-read-extra-batch-keys-need-dataexpose_-flags)) |
| `augmentation` | object | no | `{enabled, probability, ...}` |

## `model`

| Field | Type | Required | Notes |
|---|---|---|---|
| `model_type` | str | yes | A registered name; see `list-features --module models` |
| `in_channels` | int | yes | Match `dataset_type` × `coil_processing_mode` |
| `out_channels` | int | yes | Usually equals `in_channels` |
| `input_type` | enum | yes | `image`, `kspace`, `complex_image` |
| `model_kwargs` | dict | yes | Forwarded to the model constructor (e.g. `features: [32, 64, 128, 256]`). May be empty `{}` |

## `losses`

| Field | Type | Required | Notes |
|---|---|---|---|
| `policy.output_domain` | enum | yes | `image`, `kspace`, `complex`, `latent`. Flat `output_domain` folds. |
| `image_losses` | list | yes (may be empty) | Each entry: `{name, weight, enabled, ...kwargs}` |
| `kspace_losses` | list | yes (may be empty) | |
| `complex_losses` | list | yes (may be empty) | |
| `latent_losses` | list | no | Only when `output_domain == "latent"` |

Each loss entry:

```yaml
- name: l1                # registered loss name
  weight: 1.0
  enabled: true
  # any additional kwargs forwarded to the loss constructor
```

The audit's `loss_domain_block_match` check enforces that every enabled
loss's registered `domain` matches the block it appears in (with the
escape hatches documented in
[add_loss § domain-spillover](../how_to/add_loss.md#domain-spillover-when-the-audit-fights-you)).

## `optimization`

| Field | Type | Required | Notes |
|---|---|---|---|
| `optimizer.type` | enum | yes | `adamw`, `adam`, `sgd`, `rmsprop`. Flat `optimizer_type` folds. |
| `optimizer.learning_rate` | float | yes | Flat `learning_rate` folds. |
| `optimizer.weight_decay` | float | no | 0 by default. Flat `weight_decay` folds. |
| `gradient.clip.enabled` | bool | no | Flat `enable_gradient_clipping` folds. |
| `gradient.clip.value` | float | no | Flat `gradient_clip_value` folds. |
| `precision.enabled` | bool | no | Mixed precision. Most strategies are AMP-safe via `fft_ops`'s FP32 wrappers. Flat `use_amp` folds — but `amp_dtype` **raises**, it does not fold. |
| `lr_scheduler_strategy` | enum | no | `cosine`, `step`, `plateau`, … |
| `warmup_steps` | int | no | |

## `training`

| Field | Type | Required | Notes |
|---|---|---|---|
| `training_mode` | enum | yes | One of `VALID_TRAINING_MODES` |
| `strategy_class` | dotted path | yes | The corresponding entry from `STRATEGY_CLASS_PATHS` |
| `epochs` | int | yes | |
| `max_iterations` | int | no | Cap (mutually exclusive with `epochs` for some paradigms) |
| `output_dir` | path | yes | Conventional: `experiments/results/<name>` |

Two fields that used to live here have **moved and no longer fold** — the old
spellings are hard load errors, not silent relocations:

| Retired | Canonical |
|---|---|
| `training.seed` | `run.seed` |
| `training.device` | `run.device` |
| `diffusion` | object | conditional | Required for `*Diffusion*` strategies — see [gotcha 1](../how_to/write_experiment_yaml.md#1-diffusion-paradigms-need-trainingdiffusion--num_timesteps) |
| `num_timesteps` | int | conditional | Required alongside `diffusion` |

## `validation`

```yaml
validation:
  enabled: true
  scoring:
    compute: [psnr, ssim]             # registered metric names
  schedule:
    interval_steps: 5000              # iterations between val passes
  loader:
    batch_size: 2                     # may differ from the training batch size
```

`metrics:` and `eval_interval:` still fold from the flat spelling.
`val_batch_size` does **not** — it raises.

## `metrics`

```yaml
metrics:
  best_metric_name: val_psnr          # what early-stopping watches
  best_metric_mode: max               # max | min
  compute_psnr: true
  compute_ssim: true
  compute_hfen: true
  domain: image                       # which metric_compute_* hooks fire
  metric_interval: 500
```

## `undersampling`

There is no top-level `acceleration:` block — the block is called
`undersampling:` and `TrainingSettings` is `extra='forbid'`, so the old name is
a load error rather than a no-op.

```yaml
undersampling:
  acceleration_type: cartesian_vd     # which mask backend generates the pattern
  base_acceleration: 4                # 1, 2, 4, 8
  center_fraction: 0.08               # 0.04 .. 0.5
  max_acceleration: 8                 # optional cap for curriculum schedules
```

For an image-domain source, a declared acceleration is only *applied* when
`data.image_undersampling: true` is also set.

## `physics`

May be `{}` for image-domain experiments. For k-space / SENSE
experiments, declares the operator chain:

```yaml
physics:
  fft: {type: centered, norm: ortho}
  mask_generator:
    pattern: variable_density
    center_fraction: 0.08
  sensitivity:
    method: espirit                  # or 'siren_pinn'
  data_consistency: {mode: soft, weight: 0.01}
```

## Conditional blocks

Some paradigms require additional top-level blocks. The audit's
`paradigm_required_fields` check enforces them.

| Paradigm family | Required block | Fields |
|---|---|---|
| Any `*Diffusion*` | `training.diffusion` + `training.num_timesteps` | `type`, `degradation`, `num_timesteps` |
| MRF | `mrf` | `spiral_rotation_schedule`, `n_timepoints`, `sar_max_w_per_kg`, `gradient_slew_rate_t_per_m_per_s`, `phantom_calibration_path` |
| EPI distortion | `data.phase_encode_axis` | -2 or -1 |

See `src/spectramr/config/schemas/training/` for the per-paradigm sub-schemas,
and check the block name against `TrainingSettings.model_fields` before adding
one: the root schema is `extra='forbid'`, so a top-level block that does not
exist is a load error. `federated:` and `cl:` are not top-level fields.

**HPO has no config block at all.** It is entirely CLI-driven — the search
space is a separate file passed with `--search-space`, or a built-in chosen
with `--search-preset`. A top-level `hpo:` key in an experiment YAML will not
load. See [hpo_guide](../hpo_guide.md).

## Adapters (optional)

When the model's output domain disagrees with the loss's input domain,
declare the bridge:

```yaml
adapters:
  pre_loss_pred:
    - {name: ifft_kspace_to_image}
    - {name: magnitude_from_complex}
  pre_loss_target:
    - {name: ifft_kspace_to_image}
    - {name: magnitude_from_complex}
```

See [registries § adapters](registries.md#adapters--bridging-domain-mismatches)
for the list of available adapter names.

## Reporting (optional)

To auto-generate end-of-training figures:

```yaml
reporting:
  enabled: true
  task: reconstruction               # or 'super_resolution', 'synthesis'
  method_name: my_arm
  out_subdir: report
  figures:
    - fig_1_2_learning_curves
    - fig_1_3_loss_decomposition
    # paradigm-specific:
    - fig_c1_beltrami_field          # SFC / Beltrami arms
    - fig_c2_spd_geodesic            # SPD-manifold arms
    - fig_c4_fingerprint_embedding   # MRF arms
```

## Generating the JSON schema

There is no CLI flag for this — `audit` accepts `--json` (machine-readable
*findings*), which is a different thing. Generate the schema from the model:

```bash
python -c "import json
from spectramr.config.settings import TrainingSettings
print(json.dumps(TrainingSettings.model_json_schema(), indent=2))" > schema.json
```

Useful for IDE auto-completion (VS Code, JetBrains) — point the YAML
schema extension at the generated file.

## Reference template

The canonical empty-but-complete template lives at
`src/spectramr/config/schemas/templates/v1.0_reference.yaml`. Copy it
when starting a new experiment from scratch.

## Related

- [Write an experiment YAML](../how_to/write_experiment_yaml.md) — tutorial-style intro with the gotcha checklist.
- [Audit ladder](../explanation/audit_ladder.md) — what each tier enforces against this schema.
- [Registries reference](registries.md) — every registered name available in YAML.
