# Add a loss

Losses live under `src/spectramr/models/losses/`. Each loss is a callable
that takes `(pred, target, **kwargs)` and returns a scalar tensor. The
registry tracks which **domain** (image / k-space / complex / latent) the
loss expects, so the audit can reject mismatched configs at load time.

## The decorator

```python
import torch
import torch.nn as nn
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="my_loss",
    domain="image",
    aliases=["my_other_name"],     # optional
    compatible_with=None,          # optional; list of other domains
                                   # this loss is willing to consume
)
class MyLoss(nn.Module):
    """One-paragraph description.

    The math. If this is from a paper, cite it. Note any preconditions
    on the inputs (e.g. magnitude only, in [0,1], etc.).

    Args:
        alpha: Weighting parameter for the second term.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # ... loss math ...
        return loss
```

Decorator arguments:

| Argument | Notes |
|---|---|
| `name` | YAML key under `losses.image_losses[*].name`, `losses.kspace_losses[*].name`, etc. snake_case, unique. |
| `domain` | One of `"image"`, `"kspace"`, `"complex"`, `"latent"`. The audit places the loss in the matching `losses.*_losses` block. |
| `aliases` | Additional names that resolve to the same class. Useful for renames without breaking old YAMLs. |
| `compatible_with` | List of other domains the loss accepts. By default a loss is only valid in its native block. Setting `compatible_with=["image"]` lets a `domain="latent"` loss appear in `image_losses`. |

## Where to put the file

| Family | Suggested location |
|---|---|
| Pixel-level (L1, L2, Charbonnier, SSIM, perceptual) | `models/losses/image_losses.py` (group siblings) |
| k-space (NMSE, magnitude, phase) | `models/losses/kspace_losses.py` |
| Complex-valued (Hermitian, log-magnitude, phase-coherent) | `models/losses/complex_losses.py` |
| Physics-residual (data-consistency, PINN, B0 / B1 ) | `models/losses/physics_losses.py` |
| Adversarial | `models/losses/adversarial.py` |
| Distillation | `models/losses/distillation.py` |
| Latent / intertwining | `models/losses/spectral_triple_loss.py`, `latent_losses.py` |

Slop discipline: prefer one file per cohort of related losses rather than
one file per class. `models/losses/image_losses.py` hosts ~15 image-domain
losses; that's the pattern.

## Mounting in `losses/__init__.py`

Same "silent failure" trap as models. The decorator only fires when the
module is imported.

```python
# src/spectramr/models/losses/__init__.py
from . import (
    image_losses,                     # noqa: F401
    kspace_losses,                    # noqa: F401
    complex_losses,                   # noqa: F401
    spectral_triple_loss,             # noqa: F401
    your_new_loss_file,               # noqa: F401  (add this)
)
```

The no-silent-fallbacks rule is exactly this
case: a `@register_loss` decorator that nothing imports is silently dead.

A cold-subprocess regression test in
[`tests/unit/registration/test_breakthrough_components_registered.py`](https://github.com/adnaneGdihi/spectramr/blob/main/tests/unit/registration/test_breakthrough_components_registered.py)
walks every advertised loss alias and asserts it resolves. Add your new
name to that test's parameter list.

## Using your loss from a YAML

```yaml
losses:
  output_domain: image
  image_losses:
    - {name: my_loss, weight: 1.0, enabled: true, alpha: 0.5}
  kspace_losses: []
  complex_losses: []
```

The `alpha: 0.5` key is forwarded as a constructor kwarg. Any kwarg your
loss's `__init__` accepts can be set this way.

## Domain spillover (when the audit fights you)

The audit check `loss_domain_block_match` enforces that a loss appears in
the block matching its registered domain. Two ways to handle a legitimate
cross-domain use:

### Option A — declare compatibility

If the loss is genuinely safe in multiple domains, declare it:

```python
@register_loss(
    name="ssim",
    domain="image",
    compatible_with=["complex"],     # SSIM on magnitude is fine
)
class SSIMLoss(nn.Module):
    ...
```

Then a YAML can place `ssim` under `complex_losses` and pass audit.

### Option B — set `losses.output_domain`

Some paradigms produce latent vectors and feed them through latent-domain
losses. Set the parent block's `output_domain`:

```yaml
losses:
  output_domain: latent
  latent_losses:
    - {name: physics_equivariance, weight: 1.0, enabled: true}
```

The audit then expects all enabled losses to claim `domain="latent"` (or
to list `latent` in their `compatible_with`).

### Option C — explicit adapter

If your model outputs k-space but the loss expects image, declare the
bridge explicitly:

```yaml
adapters:
  pre_loss_pred:
    - {name: ifft_kspace_to_image}
    - {name: magnitude_from_complex}
  pre_loss_target:
    - {name: ifft_kspace_to_image}
    - {name: magnitude_from_complex}
```

The adapter chain runs before each loss is computed. See
`src/spectramr/data/adapters/registry.py` for the available bridges.

## Worked examples in the repo

| File | Pattern |
|---|---|
| `models/losses/image_losses.py` | The canonical pattern: 15-ish small classes registered to `domain="image"` |
| `models/losses/spectral_triple_loss.py` | Three losses sharing math (`spectral_triple_intertwining`, `connes_intertwining`, `morita_morphism`), `domain="latent"` with `compatible_with=["image"]` |
| `models/losses/physics/data_consistency.py` | k-space data consistency: `domain="kspace"` |
| `models/losses/adversarial/wgan_gp.py` | Adversarial loss with gradient penalty |

## Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| `LossRegistry.create('my_loss')` returns `None` | Module not imported by `losses/__init__.py` | Add the `# noqa: F401` import |
| Audit error `loss_domain_block_match` | Loss in wrong block | Move to matching block, set `compatible_with`, or change `output_domain` |
| `TypeError: __init__() got unexpected keyword 'alpha'` | YAML passes an arg the loss doesn't accept | Check the kwarg name in the YAML matches the constructor |
| Loss is silently 0 | `pred.detach()` or `with torch.no_grad():` accidentally inside `forward` | Remove |
| Adversarial loss explodes | Forgot gradient penalty / not using `WGAN-GP` variant | Use the WGAN-GP loss family |

## Next steps

- [Add a model](add_model.md) — the registration pattern is similar.
- [Add a paradigm](add_paradigm.md) — how to use multiple losses in one training loop.
- [Losses reference](../losses_reference.rst) — the registered loss catalogue.
