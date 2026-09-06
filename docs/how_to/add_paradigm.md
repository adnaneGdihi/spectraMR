# Add a training paradigm

A "paradigm" in spectraMR is a complete training loop: how a batch becomes a
loss, how the loss back-propagates, what gets logged, when validation runs.
Examples already in the framework: GAN, diffusion (cold / score / Lévy /
resetting), VAE, VQ-VAE, MAE / SSL, reconstruction, domain adaptation,
PINN, disentangled, sensitivity-estimation, cycle-Bloch.

Adding a new paradigm means writing a `BaseTrainingStrategy`
(see {doc}`../strategies_reference`)
subclass, registering it in three coordinated places, and landing a YAML +
test. This page is the canonical recipe.

## The four-step recipe

### 1. Write the strategy

Create a new file under
`src/spectramr/infrastructure/training/strategies/<your_paradigm>.py`:

```python
"""<one-line description of what this paradigm does>."""

from __future__ import annotations

from typing import Any

import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.strategies.base import (
    BaseTrainingStrategy,
)


class MyParadigmStrategy(BaseTrainingStrategy):
    """One-paragraph summary of the paradigm — the math, the data flow.

    Cite the paper / preprint that introduces the method if one exists.
    Note any preconditions on the data (paired? complex k-space? 3D?).
    """

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        models: dict[str, torch.nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
        losses: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """One optimisation step. Return a dict of scalar loss tensors.

        The base class's training loop pulls ``loss_total`` out of this
        dict and calls ``.backward()``. Any other keys in the return
        dict are logged.
        """
        x = batch["input"]
        target = batch["target"]
        pred = models["generator"](x)
        loss_total = losses["l1"](pred, target)
        optimizers["opt_g"].zero_grad(set_to_none=True)
        loss_total.backward()
        optimizers["opt_g"].step()
        return {"loss_total": loss_total, "loss_l1": loss_total.detach()}

    def validate_step(
        self,
        batch: dict[str, torch.Tensor],
        models: dict[str, torch.nn.Module],
        losses: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            x = batch["input"]
            target = batch["target"]
            pred = models["generator"](x)
            loss = losses["l1"](pred, target)
        return {"val_loss": loss, "pred": pred, "target": target}
```

Key conventions enforced by the audit + reviewed in PR:

- **No `.item()`, `.cpu()`, `.tolist()`, `.numpy()` in `train_step`.** They
  synchronise the GPU and kill throughput. Return tensors; the logger
  collects them async.
- **`optimizer.zero_grad(set_to_none=True)`.** Bool flag, not the default.
- **No `torch.cuda.empty_cache()` in hot paths.**
- **Return a `loss_total` key** that the base class can `.backward()` on.

### 2. Register the dispatch key

Open
[`src/spectramr/infrastructure/training/strategy_factory.py`](https://github.com/adnaneGdihi/spectramr/blob/main/src/spectramr/infrastructure/training/strategy_factory.py)
and add an entry to `STRATEGY_CLASS_PATHS`:

```python
STRATEGY_CLASS_PATHS = {
    # ... existing entries ...
    "my_paradigm": (
        "spectramr.infrastructure.training.strategies."
        "my_paradigm.MyParadigmStrategy"
    ),
}
```

Then open
[`src/spectramr/config/validation_constants.py`](https://github.com/adnaneGdihi/spectramr/blob/main/src/spectramr/config/validation_constants.py)
and add matching entries to **both**:

- `VALID_TRAINING_MODES` (set of allowed `training.training_mode` values)
- `TRAINING_MODE_CONSTRAINTS` (per-mode required fields)

```python
VALID_TRAINING_MODES = frozenset({..., "my_paradigm"})

TRAINING_MODE_CONSTRAINTS = {
    ...,
    "my_paradigm": {
        # any field you require — e.g. paired data:
        "data.dataset_type": {"nifti_paired"},
    },
}
```

**Why three places?**
- `STRATEGY_CLASS_PATHS` does runtime import — Pydantic doesn't see it.
- `VALID_TRAINING_MODES` lives in the Pydantic v2 schema and rejects
  unknown YAML values at load time.
- `TRAINING_MODE_CONSTRAINTS` declares additional cross-field invariants
  the audit ladder enforces.

A regression test ensures these three lists never drift. See
[`tests/unit/registration/test_breakthrough_components_registered.py`](https://github.com/adnaneGdihi/spectramr/blob/main/tests/unit/registration/test_breakthrough_components_registered.py)
for the pattern.

### 3. Write a reference YAML

Place it under `experiments/inprogress/<paradigm>/<arm>.yaml`:

```yaml
config_version: '1.0'

metadata:
  name: my_paradigm_baseline
  description: "Minimal reference config for my_paradigm."
  tags: {paradigm: my_paradigm, type: baseline, novelty: my_paradigm}
  version: '1.0'

data:
  dataset_type: nifti_paired

  loader:
    batch_size: 4
  sampling:
    patch_size: [256, 256, 1]
  source:
    root: ${SPECTRAMR_DATA_ROOT}/processed/ulf_to_hf_ldm/train
model:
  model_type: enhanced_deep_unet
  in_channels: 1
  out_channels: 1
  input_type: image
  model_kwargs: {features: [32, 64, 128, 256]}

losses:
  image_losses:
    - {name: l1, weight: 1.0, enabled: true}

  policy:
    output_domain: image
training:
  training_mode: my_paradigm
  strategy_class: spectramr.infrastructure.training.strategies.my_paradigm.MyParadigmStrategy
  epochs: 100
  device: cuda
  output_dir: experiments/results/my_paradigm_baseline

optimization:

  optimizer:
    type: adamw
    learning_rate: 1.0e-4
    weight_decay: 1.0e-5
validation:
  enabled: true

  scoring:
    compute: [psnr, ssim]
undersampling: {base_acceleration: 4, center_fraction: 0.08}
checkpoint: {enabled: true}
logging: {experiment_name: my_paradigm_baseline, level: info}
loss_logging: {enabled: true}
metrics: {best_metric_name: val_psnr, best_metric_mode: max, domain: image}
physics: {}
run:
  seed: 42
```

Use `${SPECTRAMR_DATA_ROOT}` for any path outside the package source tree
(see [.env.example](https://github.com/adnaneGdihi/spectramr/blob/main/.env.example)).

Validate the YAML:

```bash
spectramr audit experiments/inprogress/<paradigm>/<arm>.yaml
```

Tier 0+1 takes ~100 ms; add `--probe` for the Tier 2 synthetic forward pass.
See the [audit ladder](../explanation/audit_ladder.md) for what each tier checks.

### 4. Land tests + docs

Two files, both required by [CONTRIBUTING.md](https://github.com/adnaneGdihi/spectramr/blob/main/CONTRIBUTING.md):

```
tests/unit/infrastructure/training/test_my_paradigm_strategy.py
docs/how_to/<your_topic>.md  (or extend an existing page)
```

The unit test should at minimum:

- Resolve the strategy from `STRATEGY_CLASS_PATHS`.
- Build the strategy with a `TrainingSettings` fixture and confirm
  `train_step` produces a `loss_total` tensor.
- Confirm `training_mode='my_paradigm'` round-trips through
  `TrainingSettings.from_yaml`.

A worked-out example is
[`tests/unit/infrastructure/training/test_breakthrough_strategy_aliases.py`](https://github.com/adnaneGdihi/spectramr/blob/main/tests/unit/infrastructure/training/test_breakthrough_strategy_aliases.py).

## Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'my_paradigm'` at runtime | Forgot `STRATEGY_CLASS_PATHS` entry | Step 2 |
| `ValidationError: training_mode='my_paradigm' is not one of VALID_TRAINING_MODES` | Forgot `validation_constants` entry | Step 2 |
| Audit fires `paradigm_required_fields` even when the field is set | Forgot `TRAINING_MODE_CONSTRAINTS` entry | Step 2 |
| Strategy never runs even though audit passes | Decorator-only registry; the package `__init__.py` doesn't import the strategy file | Add an explicit import in `src/spectramr/infrastructure/training/strategies/__init__.py` |
| `loss_total` is `None` after `train_step` | Returned a plain dict without that key | Return `{"loss_total": loss, ...}` |
| Throughput drops 10× after adding paradigm | `.item()` or `.cpu()` inside `train_step` | Move to `validate_step` or async logging |

## Next steps

- [Add a model](add_model.md) — how to register a new architecture for use in this paradigm.
- [Add a loss](add_loss.md) — how to register a new loss function.
- [Write an experiment YAML](write_experiment_yaml.md) — the v6.0 schema reference.
