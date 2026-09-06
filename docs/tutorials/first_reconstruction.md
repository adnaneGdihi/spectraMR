# Your first reconstruction

This tutorial walks through a complete reconstruction experiment from
nothing to a trained model: load a synthetic phantom, undersample it in
k-space, train a U-Net to recover the missing data, and evaluate PSNR /
SSIM. It runs in **under five minutes on CPU** so you can complete it
without a GPU.

By the end you'll have:

- A registered model running through the framework's data → model →
  loss → optimizer pipeline.
- A working YAML you can use as a starting point for your own
  experiments.
- A trained checkpoint plus a PSNR / SSIM curve.

If you haven't done so yet, read [Quickstart](quickstart.md) first to
confirm `pip install spectramr[mri]` worked.

## What we're building

Forward model:

$$y = M F x + n$$

- $x$ is the ground-truth image (a synthetic phantom).
- $F$ is the centred 2D FFT (`spectramr.infrastructure.physics.fft_ops.fft2c`).
- $M$ is a Cartesian undersampling mask at 4× acceleration.
- $n$ is additive Gaussian noise.
- $y$ is the observed undersampled k-space.

We train a U-Net to learn $x = G(M^T F^{-1} y)$ — i.e. recover the
ground truth from the zero-filled IFFT of the undersampled k-space.

## Step 1 — Generate a tiny synthetic dataset

Save this as `examples/build_phantom_dataset.py`:

```python
"""Build a 64-sample synthetic phantom dataset for the tutorial.

Each sample is a 64×64 grayscale phantom: a centred Gaussian bump on
random low-amplitude noise. Saved as a single .npz file so we can use
the `npy_slice` dataset_type without filesystem setup.
"""

from pathlib import Path

import numpy as np


def make_phantom(side: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    """A toy Shepp-Logan-like phantom."""
    rng = rng or np.random.default_rng()
    y, x = np.mgrid[-1:1:side*1j, -1:1:side*1j]
    bump = np.exp(-(x**2 + y**2) / 0.1)
    noise = rng.normal(0, 0.02, (side, side))
    blob1 = 0.5 * np.exp(-((x - 0.3)**2 + (y - 0.1)**2) / 0.02)
    blob2 = -0.3 * np.exp(-((x + 0.2)**2 + (y - 0.4)**2) / 0.015)
    return (bump + blob1 + blob2 + noise).astype(np.float32)


def main() -> None:
    rng = np.random.default_rng(42)
    n_train, n_val = 48, 16
    train = np.stack([make_phantom(rng=rng) for _ in range(n_train)])
    val = np.stack([make_phantom(rng=rng) for _ in range(n_val)])
    out_dir = Path("databases/tutorial_phantom")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "train.npz", x=train)
    np.savez(out_dir / "val.npz", x=val)
    print(f"Wrote {n_train} train + {n_val} val phantoms to {out_dir}/")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python examples/build_phantom_dataset.py
```

You should see:

```
Wrote 48 train + 16 val phantoms to databases/tutorial_phantom/
```

## Step 2 — Write the experiment YAML

Save this as `experiments/inprogress/tutorials/first_reconstruction.yaml`:

```yaml
config_version: '1.0'

metadata:
  name: tutorial_first_reconstruction
  description: |
    First-reconstruction tutorial. Tiny U-Net trained on 64×64
    synthetic phantoms with 4x Cartesian undersampling.
  tags:
    paradigm: reconstruction
    type: tutorial
    novelty: tutorial_first_reconstruction
  version: '1.0'

undersampling:
  base_acceleration: 4
  center_fraction: 0.08

artifacts:
  persistent_root: experiments/results/tutorial_first_reconstruction

checkpoint:
  enabled: true
  keep_best_n: 1
  keep_last_n: 1
  save_interval: 100

data:
  dataset_type: npy_slice
  # Synthesise the aliased input from the fully-sampled phantom:
  # fft2c -> Cartesian mask (undersampling: above) -> ifft2c. This is the
  # y = M F x forward model; without it the undersampling block applies to
  # nothing and the arm silently trains at 1x.
  image_undersampling: true

  loader:
    batch_size: 8
    num_workers: 0
  coils:
    # Identity for image-domain input; keeps the phantom at 1 channel.
    processing_mode: rss_image
  sampling:
    patch_size: [64, 64, 1]
  source:
    root: ${SPECTRAMR_DATA_ROOT}/tutorial_phantom
logging:

  identity:
    experiment: tutorial_first_reconstruction
  sinks:
    level: info
loss_logging:
  enabled: true
  csv_path: experiments/results/tutorial_first_reconstruction/losses.csv

losses:
  image_losses:
    - {name: l1,   weight: 1.0, enabled: true}
    - {name: ssim, weight: 0.2, enabled: true}
  kspace_losses: []
  complex_losses: []

  policy:
    output_domain: image
metrics:
  best_metric_name: val_psnr
  best_metric_mode: max
  compute_psnr: true
  compute_ssim: true
  domain: image

model:
  model_type: enhanced_deep_unet
  in_channels: 1
  out_channels: 1
  input_type: image
  model_kwargs:
    features: [16, 32, 64]   # tiny — runs on CPU

optimization:

  optimizer:
    type: adamw
    learning_rate: 1.0e-3
    weight_decay: 1.0e-5
training:
  training_mode: reconstruction
  strategy_class: spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy
  epochs: 20
  device: cpu                # change to 'cuda' if you have one
  output_dir: experiments/results/tutorial_first_reconstruction

validation:
  enabled: true

  schedule:
    interval_steps: 100
  scoring:
    compute: [psnr, ssim]
physics: {}
run:
  seed: 42
```

Two things to notice:

- **`${SPECTRAMR_DATA_ROOT}/tutorial_phantom`** — the audit's
  [path-resolver](../CLUSTER_DATA_LAYOUT.md#resolving-paths-via-spectramr_data_root)
  expands this against your local `databases/` (if the env var is
  unset, it defaults to `./databases`). The dataset lives at
  `databases/tutorial_phantom/` from Step 1.
- **`features: [16, 32, 64]`** — a deliberately tiny U-Net so the
  whole experiment finishes in a few minutes on CPU. For a real
  experiment use `[32, 64, 128, 256]`.

## Step 3 — Validate the YAML before training

Always run the audit first. It's ~100ms and catches the bugs that
would otherwise eat a SLURM slot or a Lightning Studio session.

```bash
spectramr audit experiments/inprogress/tutorials/first_reconstruction.yaml
```

You should see something like:

```
✅ [required_section] Section 'data' present
✅ [required_section] Section 'model' present
...
✅ [model_registry] model_type='enhanced_deep_unet' registered
✅ [strategy_registry] strategy='...ReconstructionTrainingStrategy' → import-check passed
✅ [loss_domain_consistency] All loss declarations match output_domain='image'.
✅ [domain_alignment] model.in_channels=1 matches expected=1 (coil_processing_mode='rss' on image-domain dataset_type='npy_slice' collapses to 1-channel magnitude)
...
[summary] 56/56 checks passed | 0 errors | 0 warnings
```

If anything fires red, fix it before moving on. See
[audit_ladder](../explanation/audit_ladder.md) for what each check
means.

## Step 4 — Train

```bash
spectramr train --config experiments/inprogress/tutorials/first_reconstruction.yaml
```

Expected runtime: ~3–5 min on CPU, ~30 sec on GPU.

You'll see one line per training step plus periodic validation:

```
[STEP    1/120] loss_total=0.4821 loss_l1=0.4523 loss_ssim=0.149
[STEP   10/120] loss_total=0.3142 loss_l1=0.2917 loss_ssim=0.113
[STEP   50/120] loss_total=0.1855 loss_l1=0.1696 loss_ssim=0.080
[VAL  100/120] val_psnr=24.31 val_ssim=0.812
[STEP  100/120] loss_total=0.1408 loss_l1=0.1284 loss_ssim=0.062
[STEP  120/120] loss_total=0.1198 loss_l1=0.1097 loss_ssim=0.051
[VAL  120/120] val_psnr=26.07 val_ssim=0.847
[CHECKPOINT] best val_psnr=26.07 → saved to experiments/results/tutorial_first_reconstruction/checkpoints/best.pt
```

Numbers vary by seed and CPU. The important thing is that **PSNR
trends up** and **the total loss trends down**.

## Step 5 — Inspect the run

The artefact tree on disk:

```
experiments/results/tutorial_first_reconstruction/
├── checkpoints/
│   ├── best.pt              # weights at best val_psnr
│   └── last.pt              # weights at last epoch
├── logs/
│   └── train.log            # the per-step log lines
└── losses.csv               # one row per step: step, loss_total, loss_l1, loss_ssim
```

Quick sanity plot in a notebook:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("experiments/results/tutorial_first_reconstruction/losses.csv")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(df["step"], df["loss_total"], label="loss_total")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.legend()
plt.show()
```

You should see a smooth decay from ~0.5 at step 1 down to ~0.1 at
step 120.

## Step 6 — Run inference on a held-out slice

```python
import numpy as np
import torch

import spectramr  # noqa: F401  (warning + registry registration)
from spectramr.models.registry import MODEL_REGISTRY

# 1. Build the same model topology as the YAML.
entry = MODEL_REGISTRY["enhanced_deep_unet"]
model = entry["class"](in_channels=1, out_channels=1, features=[16, 32, 64])

# 2. Load the trained weights.
ckpt = torch.load(
    "experiments/results/tutorial_first_reconstruction/checkpoints/best.pt",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(ckpt["model_state"])
model.eval()

# 3. Take one val phantom and undersample it.
data = np.load("databases/tutorial_phantom/val.npz")
x = torch.from_numpy(data["x"][0:1])           # [1, 64, 64]
x = x.unsqueeze(1)                              # [1, 1, 64, 64]

# 4. Zero-filled inverse (the same operation the data pipeline applied
#    during training).
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.sampling import KSpaceMaskGenerator, MaskType

mask_gen = KSpaceMaskGenerator(
    mask_type=MaskType.CARTESIAN,
    acceleration=4,
    center_fraction=0.08,
    shape=(64, 64),
)
mask = mask_gen.generate_mask((64, 64), seed=42)        # [64, 64] real
kspace = fft2c(x.to(torch.complex64))                    # [1, 1, 64, 64] complex
kspace_us = kspace * mask
zf = ifft2c(kspace_us).abs()                             # zero-filled magnitude

# 5. Run the model.
with torch.no_grad():
    pred = model(zf)

# 6. Compute PSNR.
mse = ((pred - x) ** 2).mean().item()
psnr = 10 * np.log10(1.0 / mse)
print(f"PSNR (zero-filled vs ground truth): {10 * np.log10(1.0 / ((zf - x) ** 2).mean().item()):.2f}")
print(f"PSNR (model output vs ground truth): {psnr:.2f}")
```

Expected output (varies by seed):

```
PSNR (zero-filled vs ground truth):    19.4 dB
PSNR (model output vs ground truth):   26.1 dB
```

A ~7 dB gain over the zero-filled baseline. The exact number depends
on the seed and your CPU's FP determinism, but the trend should hold.

## What you learned

- **The framework's training loop is YAML-driven.** No Python script
  per experiment — one YAML, one command.
- **The audit ladder is your friend.** A 100ms check before
  `spectramr train` saves hours of cluster time.
- **Registries decouple component declaration from use.** You picked
  `enhanced_deep_unet` and `l1` / `ssim` by name; their actual
  implementations are in
  `src/spectramr/models/generators/` and
  `src/spectramr/models/losses/`. You never imported either.
- **The physics SSOT is the only place FFTs live.** `fft2c` /
  `ifft2c` handle centring + ortho normalisation; you never call
  `torch.fft.fft2` directly.

## Next steps

- [Write an experiment YAML](../how_to/write_experiment_yaml.md) — the full v6.0 schema, with gotchas.
- [Add a model](../how_to/add_model.md) — register your own U-Net variant.
- [Add a paradigm](../how_to/add_paradigm.md) — write a custom training loop.
- [Audit ladder](../explanation/audit_ladder.md) — what every check actually verifies.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Data root not found: tutorial_phantom` | Step 1 not run, or `SPECTRAMR_DATA_ROOT` unset and CWD isn't repo root |
| `ValidationError: training.training_mode` | Typo in the YAML — must be exactly `reconstruction` |
| `model_type='enhanced_deep_unet' not registered` | `import spectramr` not run before the registry lookup |
| `RuntimeError: CUDA out of memory` | Change `training.device: cpu` or reduce `data.batch_size` |
| `val_psnr` stays at NaN | Numerical instability — check `losses.csv`, retry with `optimization.learning_rate: 1.0e-4` |
| Training appears to hang at step 0 | Synthetic dataset loaded but worker spawn issue — set `data.num_workers: 0` in the YAML |
