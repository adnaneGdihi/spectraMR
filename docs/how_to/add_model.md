# Add a model

Models live under `src/spectramr/models/`. The framework distinguishes three
kinds:

| Kind | Directory | Purpose |
|---|---|---|
| **Generator** | `models/generators/` | A `nn.Module` whose `forward` produces the target output (image, k-space, fingerprint). |
| **Discriminator** | `models/discriminators/` | A `nn.Module` for adversarial training paradigms. |
| **Block** | `models/blocks/` | A reusable building block (attention head, normalisation layer, FFT-conv). Not registered — composed by generators. |

This page covers generators (the most common case). The discriminator
pattern is identical, just with a different registry surface.

## The decorator

Every generator is registered via `@register_model` at class definition
time. Reading the decorator tells the framework what the model accepts
and produces — without it, the model is invisible to the YAML dispatch.

```python
from spectramr.models.registry import register_model

@register_model(
    name="my_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    accepts_complex=False,
    expects_real_imag_interleaved=False,
    supports_contrast_conditioning=False,
)
class MyUNet(nn.Module):
    """One-paragraph summary of what this architecture does.

    Args:
        in_channels: Input image channels.
        out_channels: Output image channels.
        features: Width per encoder/decoder stage.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
    ) -> None:
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

Decorator arguments:

| Argument | Type | Notes |
|---|---|---|
| `name` | `str` | The YAML key (`model.model_type`). Must be unique. snake_case. |
| `training_mode` | `str` | Default paradigm this model is built for. The audit warns if a YAML pairs the model with an incompatible mode. |
| `spatial_dims` | `tuple[int, ...]` | Which spatial dimensionalities the forward pass accepts. `(2,)` = 2D only. `(2, 3)` = both. `(1,)` = fingerprint / time-series. The audit's `data_model_compatibility` check uses this. |
| `accepts_complex` | `bool` | True if `forward` receives `torch.complex64` tensors directly. |
| `expects_real_imag_interleaved` | `bool` | True if the model wants `[B, 2C, H, W]` where each complex channel is split into real/imag pairs (the typical real-valued encoding of k-space). |
| `supports_contrast_conditioning` | `bool` | True if the model accepts a contrast embedding as a second argument. |

## Where to put the file

| Architecture family | Suggested location |
|---|---|
| U-Net variant | `models/generators/<descriptive_name>_unet.py` |
| Diffusion network | `models/generators/diffusion/<name>.py` |
| Attention-bottleneck | `models/generators/breakthrough_attention_generators.py` (group siblings together) |
| Geometric prior | `models/generators/breakthrough_geometric_generators.py` |
| Reconstruction (unrolled, ISTA, ADMM) | `models/generators/unrolled/<name>.py` |

The convention is *minimum slop*: bundle siblings in one file when their
math is closely related, not one-class-per-file. The
`breakthrough_attention_generators.py` file in the repo hosts five attention
variants in a single 800-line module — that's the model you want.

## Mounting in `models/__init__.py`

This is the **silent failure** trap. A `@register_model` decorator only
fires when its module is imported. The package's `__init__.py` must import
the file for cold-import correctness:

```python
# src/spectramr/models/generators/__init__.py
from . import (
    breakthrough_attention_generators,    # noqa: F401  (registers 5 models)
    breakthrough_geometric_generators,    # noqa: F401  (registers 2 models)
    your_new_model_file,                  # noqa: F401  (registers MyUNet)
)
```

Without this, your model is dead — `MODEL_REGISTRY.get("my_unet")` returns
`None`, and the YAML audit fails with `model_type='my_unet' not registered`.

A cold-subprocess regression test catches this. Pattern:

```python
import subprocess, sys
def test_my_unet_registered_cold():
    rc = subprocess.run(
        [sys.executable, "-c",
         "import spectramr.models.generators; "
         "from spectramr.models.registry import MODEL_REGISTRY; "
         "assert MODEL_REGISTRY.get('my_unet') is not None"],
        capture_output=True, text=True, timeout=60,
    ).returncode
    assert rc == 0
```

See [`tests/unit/registration/test_breakthrough_components_registered.py`](https://github.com/adnaneGdihi/spectramr/blob/main/tests/unit/registration/test_breakthrough_components_registered.py)
for the canonical pattern.

## What gets stored in the registry

After registration, `MODEL_REGISTRY["my_unet"]` is a dict:

```python
{
    "class": MyUNet,                       # the class itself
    "mode": "reconstruction",              # the default training_mode
    "supports_contrast_conditioning": False,
    "capabilities": ModelCapabilities(
        spatial_dims=(2,),
        input_domain=None,                 # set explicitly if your model
        output_domain=None,                # consumes/produces a specific
        accepts_complex=False,             # domain — used by adapter
        expects_real_imag_interleaved=False,
        requires_paired_data=None,
    ),
}
```

The instance you get back from `entry["class"](in_channels=..., **kwargs)` is
a plain `nn.Module`. The YAML's `model.model_kwargs` block becomes the
kwargs.

## Worked examples in the repo

| File | What it shows |
|---|---|
| `models/generators/breakthrough_attention_generators.py` | Five small U-Nets that wrap different attention blocks. The pattern: a stack-of-blocks generator that delegates the interesting math to `models/blocks/`. |
| `models/generators/breakthrough_geometric_generators.py` | Hyperbolic + Heisenberg generators. Shows how to add a bottleneck operating on a non-Euclidean manifold. |
| `models/generators/diffusion/` | Score-network architectures (U-Net + time embedding). |
| `models/generators/complex_unet.py` | Complex-valued generator with `accepts_complex=True`. |
| `models/generators/mrf_models.py` | 1D fingerprint operator with `spatial_dims=(1,)`. |

## Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| `MODEL_REGISTRY.get('my_unet') is None` | Module not imported by `__init__.py` | Add the import (with `# noqa: F401`) |
| `data_model_compatibility` audit failure | `spatial_dims` doesn't match the data's patch dimensionality | Either fix `spatial_dims=` to match, or declare an `adapters.pre_model:` chain in the YAML |
| `expected even channel count` audit error | Model declares `expects_real_imag_interleaved=True` but YAML has odd `in_channels` | Channel count must be `2 * complex_channels` |
| Throughput drops after wrapping in `MyUNet` | Forward pass calls `.item()` for shape debugging | Use `tensor.shape` (no GPU sync) |

## Next steps

- [Add a paradigm](add_paradigm.md) — pair the new model with a custom training loop.
- [Model registry reference](../model_registry_reference.rst) — the registered model catalogue.
