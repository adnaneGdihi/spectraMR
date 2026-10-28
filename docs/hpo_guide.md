# HPO Guide

Hyperparameter optimization in this repo is a thin layer over Optuna with three things bolted on for research-friendliness:

1. **Subprocess-isolated trials** — each trial spawns its own `python -m spectramr.cli train` so a single trial crash (NaN, OOM, lib mismatch) cannot poison the parent study's TPE state.
2. **Auto-write-back of `best_config.yaml` + `best_params.json`** at study completion — the winning hyperparameters land as a runnable YAML you can directly feed into `train --config best_config.yaml`.
3. **Schema-aware search spaces** — every dotted-path field in `TrainingSettings` is targetable, including per-loss weights via `[name=foo]` list-selector syntax. `--list-schema-paths` prints the full set; named presets cover the common cases.

## TL;DR

```bash
# Sweep learning rate and weight decay on a shipped baseline arm
python -m spectramr.cli hpo \
    --config experiments/inprogress/workflow_baselines/b1_structural_recon_m4raw.yaml \
    --model-type complex_unet \
    --search-preset optimizer_basic \
    --n-trials 40 \
    --max-iter 30000 \
    --storage sqlite:///experiments/hpo/optimizer.db

# Output: <training.output_dir>/hpo/hpo_complex_unet/
#   ├── best_config.yaml   ← rerun training with this
#   ├── best_params.json   ← raw Optuna params + objective value
#   └── trial_NNNN/        ← per-trial configs + checkpoints + metrics

# Train the winner directly
python -m spectramr.cli train --config <output_dir>/best_config.yaml
```

`--model-type` names the study and its output directory; it does **not** select
the model. Each trial trains whatever `model.model_type` the config declares, so
pass the config's own value or the output directory will be mislabelled.

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
| `kan_dual_domain` | 12 | LR, KAN grid/order/hidden, radial bands, 4 loss weights, curriculum |
| `loss_weights_only` | 4 | Just the 4 composite-loss weights (use after picking a fixed architecture) |
| `optimizer_basic` | 2 | LR + weight decay (use after picking architecture + loss weights) |
| `full_kan_block` | 8 | Every KAN block hyperparameter (deeper sweep than `kan_dual_domain`) |
| `diffusion_curriculum` | 3 | Curriculum ramp + identity-collapse threshold |

Pick one with `--search-preset <name>`.

`optimizer_basic` applies to any config. `diffusion_curriculum` needs a diffusion
arm, which declares the curriculum fields it targets. The other three target
`losses.kspace_losses[name=...]` entries and
`model.model_kwargs.kan_dual_domain_kwargs.*`: a list selector raises
`no list item with name=...` on trial 1 if the config does not already declare
that loss, so check what your arm declares before picking one.

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
# Each top-level key is a dotted-path config field. Use the canonical nested
# spelling.
optimization.optimizer.learning_rate:
  dist: loguniform        # or uniform, int_uniform, int_loguniform, categorical
  low: 1.0e-5
  high: 2.0e-4

optimization.optimizer.weight_decay:
  dist: loguniform
  low: 1.0e-7
  high: 1.0e-3

# List-selector syntax for losses: targets the entry with name=l1
losses.kspace_losses[name=l1].weight:
  dist: loguniform
  low: 0.01
  high: 1.0

# Categorical with explicit choices
training.batch_size:
  dist: categorical
  choices: [2, 4, 8]

# Free-form dict fields (model_kwargs etc.) accept any sub-key
model.model_kwargs.some_new_flag:
  dist: categorical
  choices: [true, false]
```

Distributions supported: `uniform(low, high)`, `loguniform(low, high)`, `int_uniform(low, high)`, `int_loguniform(low, high)`, `categorical(choices)`.

## Worked examples

### Tune a loss weight with a custom search space

The built-in loss-weight presets name losses a config must already declare. To
tune the losses *your* arm declares, write the search space yourself — the
baseline arm above declares one `kspace_losses` entry, `l1`:

```yaml
# my_space.yaml
losses.kspace_losses[name=l1].weight:
  dist: loguniform
  low: 0.1
  high: 10.0
optimization.optimizer.learning_rate:
  dist: loguniform
  low: 1.0e-5
  high: 2.0e-4
```

```bash
python -m spectramr.cli hpo \
    --config experiments/inprogress/workflow_baselines/b1_structural_recon_m4raw.yaml \
    --model-type complex_unet \
    --search-space my_space.yaml \
    --n-trials 50 \
    --max-iter 30000 \
    --storage sqlite:///experiments/hpo/loss_weights.db
```

A selector must name the list the loss actually lives in. Plain dotted keys
auto-create missing intermediate dicts, so a `model_kwargs` path an arm never
declares is harmless; list selectors are the opposite and raise on trial 1.

### Search in two passes

Two-pass HPO is often more efficient than searching everything at once, because
TPE handles a smaller-dimensional space better. Pass 1 fixes the schedule; pass 2
tunes the optimizer against it.

```bash
# Pass 1: diffusion curriculum
python -m spectramr.cli hpo \
    --config experiments/inprogress/diffusion/experiment_96_sde_diffusion.yaml \
    -m score_based_diffusion \
    --search-preset diffusion_curriculum \
    --n-trials 60 \
    --storage sqlite:///experiments/hpo/pass1.db

# Pass 2: optimizer, starting from Pass 1's best_config.yaml
python -m spectramr.cli hpo \
    --config <pass1_output>/hpo/hpo_score_based_diffusion/best_config.yaml \
    -m score_based_diffusion \
    --search-preset optimizer_basic \
    --n-trials 40 \
    --storage sqlite:///experiments/hpo/pass2.db
```

Pass 2 starts from Pass 1's winning config, so the schedule is fixed and only the
optimizer varies.

### Multi-objective: PSNR vs training time

```bash
python -m spectramr.cli hpo \
    --config <yaml> -m <model> \
    --search-preset optimizer_basic \
    --multi-objective \
    --cost-weight 0.3 \
    --objective-metric val_psnr \
    --n-trials 60 \
    --storage sqlite:///experiments/hpo/pareto.db
```

When `--multi-objective` is set with `--cost-weight > 0`, the optimizer treats `(objective_metric, training_time_seconds)` as a Pareto-front search. The best-config writer picks the highest-PSNR Pareto point but the JSON has the full front for further analysis.

## Resumable + parallel studies

The `--storage sqlite:///path.db` URL makes the study resumable across machine reboots and parallelizable across worker processes. To resume after a crash:

```bash
# Same command, same study name (defaults to model_type), --load-if-exists is implicit
python -m spectramr.cli hpo --config <yaml> -m <model> \
    --search-preset optimizer_basic \
    --n-trials 30 \
    --storage sqlite:///experiments/hpo/optimizer.db
```

To parallelize across nodes (each worker contributes additional trials to the same study):

```bash
# On each node — same storage URL, same model_type, fresh n-trials per worker
python -m spectramr.cli hpo --config <yaml> -m <model> --search-preset optimizer_basic \
    --n-trials 20 --storage sqlite:///shared/hpo.db
```

NFS-backed SQLite works for ≤4 workers; for more, consider PostgreSQL via `--storage postgresql://...`.

## Pruning behavior

Default pruner is Hyperband with intermediate reports at iters 4K / 8K / 16K / 32K. To switch:

```bash
python -m spectramr.cli hpo --config <yaml> -m <model> \
    --search-preset optimizer_basic \
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

* **A search-space path must be the one the schema declares.** A search space *adds* a key to a config that already carries one, so a path that is merely close raises on trial 1 rather than being quietly accepted. Write the full nested path — `optimization.optimizer.learning_rate`, not a shorter spelling — and check it against the schema before running.

* **Wavelet-attention configs cannot use the `kan_dual_domain` preset directly.** The preset defines paths inside `model.model_kwargs.kan_dual_domain_kwargs` which the wavelet attention type ignores. For wavelet HPO, write a custom YAML targeting the wavelet-specific paths (e.g., `num_levels`, `score_fn`).

* **A `[name=...]` selector must name the list the loss actually lives in.** Plain dotted keys auto-create missing intermediate dicts, so a `model_kwargs` path an arm never declares is harmless. List selectors are the opposite: they raise `no list item with name=...` and kill the run on trial 1. The same loss name can be declared under `losses.kspace_losses` or `losses.image_losses` depending on the arm, and the selector does not search across both — read the config's `losses:` block and write the list it uses.

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
    "optimization.optimizer.weight_decay": ("loguniform", 1e-7, 1e-3),
    "losses.kspace_losses[name=l1].weight": ("loguniform", 0.1, 10.0),
})

# Run HPO
request = HPORequest(
    config_path="experiments/inprogress/workflow_baselines/b1_structural_recon_m4raw.yaml",
    model_types=["complex_unet"],
    n_trials=60,
    objective_metric="val_psnr",
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

The HPO machinery's unit tests cover distribution validation, search-space construction (dict / YAML / programmatic), the preset registry, composite merging with conflict policies, dotted-path tokenization (including `[name=foo]` list selectors), schema enumeration, and end-to-end application of a preset to a trial YAML. Run them locally:

```bash
pytest tests/unit/pipelines/test_hpo_search_spaces.py \
       tests/unit/infrastructure/coordination/test_hpo_coordinator.py \
       tests/unit/application/use_cases/test_hpo_use_case.py \
       tests/unit/pipelines/test_hpo_subprocess_objective.py
```

The coordinator tests use `MagicMock` for `EnhancedHPOptimizer` and `subprocess_training_objective`, so they run on CPU in seconds without spawning any real trainers.
