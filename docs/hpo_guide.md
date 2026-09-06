# HPO Guide

Hyperparameter optimization in this repo is a thin layer over Optuna with three things bolted on for research-friendliness:

1. **Subprocess-isolated trials** — each trial spawns its own `python -m spectramr.cli train` so a single trial crash (NaN, OOM, lib mismatch) cannot poison the parent study's TPE state.
2. **Auto-write-back of `best_config.yaml` + `best_params.json`** at study completion — the winning hyperparameters land as a runnable YAML you can directly feed into `train --config best_config.yaml`.
3. **Schema-aware search spaces** — every dotted-path field in `TrainingSettings` is targetable, including per-loss weights via `[name=foo]` list-selector syntax. There are 1176 dotted paths total; 5 named presets cover the common cases.

## TL;DR

```bash
# Most common: search the plan §4.2 KAN dual-domain space
python -m spectramr.cli hpo \
    --config experiments/inprogress/kspace_filling/experiment_11_kan_dual_domain.yaml \
    --model-type kspace_cold_diffusion \
    --search-preset kan_dual_domain \
    --n-trials 80 \
    --max-iter 30000 \
    --storage sqlite:///experiments/hpo/kan.db

# Output: experiments/results/.../hpo/hpo_kspace_cold_diffusion/
#   ├── best_config.yaml   ← rerun training with this
#   ├── best_params.json   ← provenance for the paper
#   └── trial_NNNN/        ← per-trial configs + checkpoints + metrics

# Train the winner directly
python -m spectramr.cli train --config <output_dir>/best_config.yaml
```

## Three ways to specify the search space

Pick the *least invasive* mechanism that fits your needs.

### 1. Built-in preset (terse)

5 named presets ship with the repo:

```bash
# --config and --model-type are required even though this flag reads neither;
# any loadable path and any registered model_type will do.
python -m spectramr.cli hpo --config <yaml> -m <model> --list-presets
```

| Preset | Dimension | What it tunes |
|---|---|---|
| `kan_dual_domain` | 12 | Plan §4.2 — LR, KAN grid/order/hidden, radial bands, 4 loss weights, curriculum |
| `loss_weights_only` | 4 | Just the 4 composite-loss weights (use after picking a fixed architecture) |
| `optimizer_basic` | 2 | LR + weight decay (use after picking architecture + loss weights) |
| `full_kan_block` | 8 | Every KAN block hyperparameter (deeper sweep than `kan_dual_domain`) |
| `diffusion_curriculum` | 3 | Curriculum ramp + identity-collapse threshold |

Pick one with `--search-preset <name>`.

### 2. Composite presets (compose narrow ones)

`--search-preset` is repeatable. Multiple presets are merged into a single union space:

```bash
# Combine two presets
python -m spectramr.cli hpo --config <yaml> -m <model> \
    --search-preset optimizer_basic \
    --search-preset loss_weights_only
```

Conflicts (same dotted path defined in multiple presets) raise an error by default — pass `--preset-merge-policy override` (later preset wins) or `--preset-merge-policy keep` (first preset wins) to resolve explicitly. Loud failure is the default because conflicting overlaps are usually a bug.

### 3. User-authored YAML (most flexible)

For arbitrary search spaces, write your own YAML:

```bash
# Generate a starter template. --config/--model-type are required here too,
# and unread here too.
python -m spectramr.cli hpo --config <yaml> -m <model> --print-template > my_space.yaml

# List every dotted-path field TrainingSettings exposes
python -m spectramr.cli hpo --config <yaml> -m <model> --list-schema-paths

# Run with your spec
python -m spectramr.cli hpo --config <yaml> -m <model> --search-space my_space.yaml
```

YAML format:

```yaml
# Each top-level key is a dotted-path config field. Use the CANONICAL nested
# spelling -- see the pitfall about retired spellings below.
optimization.optimizer.learning_rate:
  dist: loguniform        # or uniform, int_uniform, int_loguniform, categorical
  low: 1.0e-5
  high: 2.0e-4

optimization.optimizer.weight_decay:
  dist: loguniform
  low: 1.0e-7
  high: 1.0e-3

# List-selector syntax for losses: targets the entry with name=log_spectral
losses.kspace_losses[name=log_spectral].weight:
  dist: loguniform
  low: 0.01
  high: 1.0

# Categorical with explicit choices
model.model_kwargs.kan_dual_domain_kwargs.kan_grid_size:
  dist: categorical
  choices: [4, 5, 6, 8]

# Free-form dict fields (model_kwargs etc.) accept any sub-key
model.model_kwargs.some_new_flag:
  dist: categorical
  choices: [true, false]
```

Distributions supported: `uniform(low, high)`, `loguniform(low, high)`, `int_uniform(low, high)`, `int_loguniform(low, high)`, `categorical(choices)`.

## Worked examples

### Tune just the loss weights for the headline KAN dual-domain

```bash
python -m spectramr.cli hpo \
    --config experiments/inprogress/kspace_filling/experiment_11_kan_dual_domain.yaml \
    --model-type kspace_cold_diffusion \
    --search-preset loss_weights_only \
    --n-trials 50 \
    --max-iter 30000 \
    --storage sqlite:///experiments/hpo/loss_weights.db
```

This searches the four loss weights (`log_spectral`, `sobolev_kspace`, `complex_spatial_gradient`, `sense_adjoint_l1`) over their plan-recommended log-uniform ranges. ~50 trials is enough to converge on a 4D space.

### Tune the KAN block, then the optimizer, sequentially

Two-pass HPO is often more efficient than searching everything at once because TPE handles a smaller-D space better. Pass 1: pick the right KAN architecture; Pass 2: tune optimizer for that architecture.

```bash
# Pass 1: KAN block hyperparams
python -m spectramr.cli hpo \
    --config experiments/inprogress/kspace_filling/experiment_11_kan_dual_domain.yaml \
    -m kspace_cold_diffusion \
    --search-preset full_kan_block \
    --n-trials 60 \
    --storage sqlite:///experiments/hpo/pass1.db

# Pass 2: optimizer + curriculum, starting from Pass 1's best_config.yaml
python -m spectramr.cli hpo \
    --config experiments/results/<pass1_output>/hpo/hpo_kspace_cold_diffusion/best_config.yaml \
    -m kspace_cold_diffusion \
    --search-preset optimizer_basic \
    --search-preset diffusion_curriculum \
    --n-trials 40 \
    --storage sqlite:///experiments/hpo/pass2.db
```

Pass 2 starts from Pass 1's winning config so the architecture is fixed; only the optimizer + curriculum vary.

### Use a custom YAML to search S-map FiLM intensity

Suppose you want to ablate the S-map FiLM hidden width while also tuning the KAN ADC's grid size:

```yaml
# my_smap_kan_adc.yaml
model.model_kwargs.kan_dual_domain_kwargs.smap_film_hidden:
  dist: categorical
  choices: [16, 32, 64, 128]
model.model_kwargs.kan_dc_kwargs.kan_grid_size:
  dist: categorical
  choices: [4, 5, 6, 8]
model.model_kwargs.kan_dc_kwargs.kan_hidden:
  dist: categorical
  choices: [8, 16, 24]
```

```bash
python -m spectramr.cli hpo \
    --config experiments/inprogress/kspace_filling/attention_enhancements/experiment_11_attn_kan_smap.yaml \
    -m kspace_cold_diffusion \
    --search-space my_smap_kan_adc.yaml \
    --n-trials 40 \
    --storage sqlite:///experiments/hpo/smap.db
```

### Multi-objective: PSNR vs training time

```bash
python -m spectramr.cli hpo \
    --config <yaml> -m <model> \
    --search-preset full_kan_block \
    --multi-objective \
    --cost-weight 0.3 \
    --objective-metric val_robust_mri_psnr_4x \
    --n-trials 60 \
    --storage sqlite:///experiments/hpo/pareto.db
```

When `--multi-objective` is set with `--cost-weight > 0`, the optimizer treats `(objective_metric, training_time_seconds)` as a Pareto-front search. The best-config writer picks the highest-PSNR Pareto point but the JSON has the full front for further analysis.

## Resumable + parallel studies

The `--storage sqlite:///path.db` URL makes the study resumable across machine reboots and parallelizable across worker processes. To resume after a crash:

```bash
# Same command, same study name (defaults to model_type), --load-if-exists is implicit
python -m spectramr.cli hpo --config <yaml> -m <model> \
    --search-preset kan_dual_domain \
    --n-trials 30 \
    --storage sqlite:///experiments/hpo/kan.db
```

To parallelize across nodes (each worker contributes additional trials to the same study):

```bash
# On each node — same storage URL, same model_type, fresh n-trials per worker
python -m spectramr.cli hpo --config <yaml> -m <model> --search-preset kan_dual_domain \
    --n-trials 20 --storage sqlite:///shared/hpo.db
```

NFS-backed SQLite works for ≤4 workers; for more, consider PostgreSQL via `--storage postgresql://...`.

## Pruning behavior

Default pruner is Hyperband with intermediate reports at iters 4K / 8K / 16K / 32K. To switch:

```bash
python -m spectramr.cli hpo --config <yaml> -m <model> \
    --search-preset kan_dual_domain \
    --pruner median           # or successive_halving
```

> **`--pruner none` does not disable pruning, and `--pruner threshold` does not run.**
> The CLI's `choices=` list and the factory that consumes the string are two
> unsynced vocabularies.
> `none` is translated to `nop`, the pruner factory has no `nop` branch, and the
> fall-through returns **MedianPruner** — so trials you expected to run to
> completion get pruned instead. `threshold` raises
> `TypeError: Either lower or upper must be specified.` before the first trial,
> because nothing plumbs its bounds. The pruners that behave as documented are
> `hyperband`, `median` and `successive_halving`. Likewise `--sampler nsga2` is
> matched as `nsgaii` in the factory and silently degrades to TPE unless
> `--multi-objective --cost-weight >0` is also set.

## Output layout

After completion, each model's HPO writes:

```
<output_dir>/hpo/hpo_<model_type>/
├── best_config.yaml         # base config + winning overrides; runnable
├── best_params.json         # raw Optuna params + objective value (provenance)
├── trial_0000/
│   ├── config.yaml          # the YAML for this trial
│   ├── logs/loss_log.csv    # what HPO polled for the objective
│   ├── checkpoints/         # if the trial wasn't pruned
│   └── metrics/
├── trial_0001/
└── ...
```

`best_config.yaml` annotates the metadata block with `hpo_source`, `hpo_objective_metric`, and `hpo_best_value` so the YAML is self-documenting about why these particular hyperparameters were picked.

The output directory defaults to `<base.training.output_dir>/hpo` so HPO trial artifacts never overwrite the parent training run's output. Override with `--output-dir`.

## Common pitfalls

* **No search space → no-op HPO.** If neither `--search-preset` nor `--search-space` is given, the coordinator logs a loud warning and every trial runs the unchanged base config. The "best trial" report will have an empty `best_params: {}` dict — that's the diagnostic. Always pass a search space.

* **Conflict between presets.** `--search-preset a --search-preset b` raises if `a` and `b` define the same path. This is intentional — silent precedence rules are usually a bug. If you genuinely want one to override the other, pass `--preset-merge-policy override` (later wins) or `keep` (first wins).

* **Pruning at iter 4K when `--max-iter 5000`.** The Hyperband milestones don't auto-rescale to your `--max-iter`. Pick a budget where pruning makes sense (`--max-iter 30000+`) — you cannot currently opt out, because `--pruner none` silently becomes MedianPruner.

* **`train --override` cannot express a `[name=...]` selector.** The selector syntax works in a *search space*, which resolves paths through `apply_dotted_override`. `train --override` uses a different resolver that splits on the first `=` — which lands inside the selector — and then discards the result while logging `Overrides applied (1)`. Applying a winning loss weight by hand means editing the YAML, or just running the `best_config.yaml` that HPO already wrote.

* **A retired spelling in a search space raises on trial 1.** `optimization.learning_rate` and `optimization.weight_decay` are `fold` records: a YAML that declares *only* the old spelling still loads. A search space is different, because it *adds* a key to a config that already has one. If the arm declares `optimization.optimizer.learning_rate` and the search space names `optimization.learning_rate`, both are present with different values and the fold's conflict guard raises `optimization.learning_rate=... and optimization.optimizer.learning_rate=... disagree`. Verified against `experiment_11_kan_dual_domain.yaml`, which declares the nested spelling — so both of these must be written the canonical way. Resolve any path you are unsure of through `config/schemas/renames.py` before putting it in a search space, and note that the fold is *silent when the values happen to coincide*, which makes a spot check with a round number useless as a test.

* **Wavelet-attention configs cannot use the `kan_dual_domain` preset directly.** The preset defines paths inside `model.model_kwargs.kan_dual_domain_kwargs` which the wavelet attention type ignores. For wavelet HPO, write a custom YAML targeting the wavelet-specific paths (e.g., `num_levels`, `score_fn`).

* **A `[name=...]` selector must name the list the loss actually lives in.** Plain dotted keys auto-create missing intermediate dicts, so a `model_kwargs` path an arm never declares is harmless. List selectors are the opposite: they raise `no list item with name=...` and kill the run on trial 1. Until 2026-07-30 the `kan_dual_domain` and `loss_weights_only` presets targeted `losses.image_losses[name=complex_spatial_gradient]` and `[name=sense_adjoint_l1]`, but both losses live in `losses.kspace_losses` (57 and 56 arms corpus-wide, and in `image_losses` nowhere), so both presets crashed on their first trial. When adding a preset path, check which list the arm declares the loss under — `tests/unit/pipelines/test_hpo_search_spaces.py::test_every_preset_applies_cleanly_to_headline_yaml` now applies every built-in preset to the headline arm to keep them honest.

## Programmatic API

For HPO from inside a Python script (a notebook, a custom analysis):

```python
from spectramr.application.use_cases.hpo_use_case import HPORequest, HPOUseCase
from spectramr.infrastructure.logging import LoggingService
from spectramr.pipelines.hpo_search_spaces import SearchSpace, load_preset, load_presets

# Three ways to build a search space:

# (a) Named preset
space = load_preset("kan_dual_domain")

# (b) Merge multiple presets (raises on conflict by default)
space = load_presets(["optimizer_basic", "loss_weights_only"])

# (c) Programmatic — use tuple specs for terseness or dict specs for readability
space = SearchSpace.from_dict({
    "optimization.optimizer.learning_rate": ("loguniform", 1e-5, 2e-4),
    "model.model_kwargs.kan_dual_domain_kwargs.kan_grid_size": ("categorical", 4, 5, 6, 8),
    "losses.kspace_losses[name=log_spectral].weight": ("loguniform", 0.01, 1.0),
})

# Run HPO
request = HPORequest(
    config_path="experiments/inprogress/kspace_filling/experiment_11_kan_dual_domain.yaml",
    model_types=["kspace_cold_diffusion"],
    n_trials=60,
    objective_metric="val_robust_mri_psnr_2x",
    max_iter_per_trial=30000,
    storage_url="sqlite:///my_study.db",
    search_space_dict={
        path: {
            "dist": dist.kind,
            **({"low": dist.low, "high": dist.high} if dist.kind != "categorical" else {}),
            **({"choices": dist.choices} if dist.kind == "categorical" else {}),
        }
        for path, dist in space.params.items()
    },
)
response = HPOUseCase(LoggingService()).execute(request)
print(f"Success: {response.success}")
for model_type, result in response.results.items():
    print(f"  {model_type} best_value: {result.get('best_value')}")
    print(f"  {model_type} best_config: {result.get('best_config_path')}")
```

## Testing

The HPO machinery has 37 unit tests covering: distribution validation, search-space construction (dict / YAML / programmatic), preset registry, composite merging with conflict policies, dotted-path tokenization (including `[name=foo]` list selectors), schema enumeration, and end-to-end trial-YAML application of the `kan_dual_domain` preset against the real headline YAML. Run them locally:

```bash
pytest tests/unit/pipelines/test_hpo_search_spaces.py \
       tests/unit/infrastructure/coordination/test_hpo_coordinator.py \
       tests/unit/application/use_cases/test_hpo_use_case.py \
       tests/unit/pipelines/test_hpo_subprocess_objective.py
```

The coordinator tests use `MagicMock` for `EnhancedHPOptimizer` and `subprocess_training_objective`, so they run on CPU in seconds without spawning any real trainers.
