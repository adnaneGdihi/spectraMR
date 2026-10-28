# spectraMR

> **NOT FOR CLINICAL USE.** Research software only. See [DISCLAIMER.md](DISCLAIMER.md).

[![PyPI version](https://img.shields.io/pypi/v/spectramr.svg)](https://pypi.org/project/spectramr/)
[![Python](https://img.shields.io/pypi/pyversions/spectramr.svg)](https://pypi.org/project/spectramr/)
[![License](https://img.shields.io/github/license/adnaneGdihi/spectraMR.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22291316.svg)](https://doi.org/10.5281/zenodo.22291316)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

<!-- Verify a badge by its rendered <title>, never by HTTP status: shields serves
     200 with "not found" painted into the SVG. All five above were checked that
     way on 2026-09-04 (pypi v0.1.0 / python 3.12 / license Apache-2.0).

     Held back, each with the one thing that fills it:
       CI        -- pr-required.yml triggers on pull_request, so `main` has no run
                    and shields reports the newest PR's instead.
       docs      -- the Read the Docs project is not imported yet.
       downloads -- pepy serves a 45-byte 404 and shields/pypistats answers
                    "rate limited by upstream service"; both need PyPI history.
       codecov   -- no report has been uploaded.

     The DOI is Zenodo's CONCEPT DOI, the parent that follows every release, not
     the per-deposit version DOI (which would freeze at v0.1.0). Its number and
     the line's shape are owned by scripts/release/zenodo_deposit.py and pinned
     here by tests/unit/release/test_zenodo_deposit.py. Do NOT swap in Zenodo's
     repo-id badge form: it belongs to Zenodo's GitHub integration, which only
     fires on a GitHub release of a public repo and so never registered a mapping
     for the REST-API deposits used here. Probed against this repo's numeric id,
     `1347566284`: both `/badge/1347566284.svg` and `/badge/latestdoi/1347566284`
     return 404. It is the shape that looks filled and is dead. -->

spectraMR is a research framework for MRI reconstruction, super-resolution,
quantitative mapping, and generative modelling. It registers 153 training
strategies (selectable through 206 `training_mode` spellings), 586 model
architectures, 217 losses, and a single-source-of-truth MRI physics layer
(centered FFT, Cartesian / VD / radial / NUFFT sampling masks, ESPIRiT and
SIREN-PINN coil maps, hard / soft data consistency, Bloch / motion / B0 / B1⁻
simulation).

These are **registration** counts, measured on the shipped package rather than
estimated, under `pip install spectramr[mri]` -- the install the quick start
prescribes. They are a property of the *installed extras*, not of the
distribution: a bare `pip install spectramr` registers **175** models rather than
586, because a model whose module fails to import is not registered at all
(loudly -- discovery names the module and the missing package). See
[Installation](#installation).

Every registered model, loss, metric and transform is reachable from
a cold import — 586 / 217 / 211 / 10, cold-probe equal to walk, verified by
`scripts/maintenance/prove_reachable.py --audit`; the 206 strategy paths are a
static dict and all 206 resolve. That is a reachability claim, not a validation
one: it says a config can select the component, not that the component is
benchmarked. Per-regime maturity is graded LIVE / PARTIAL / EVAL_ONLY / STUB by
the `Maturity` ledger, and `docs/known_limitations.rst` records what is known
not to work. Re-measure rather than quoting these; they move week to week.

## Quick start

```bash
pip install spectramr[mri]
```

A single forward pass through a small U-Net via the registry:

```python
import torch
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY

# Required. The registry is EMPTY on a plain import -- a model is registered
# only once the module holding its decorator has been imported, and this call
# is what curates those imports. Without it MODEL_REGISTRY.get() returns None
# and the next line raises TypeError.
populate_model_registry()

entry = MODEL_REGISTRY.get("toeplitz_attention_unet")
model = entry["class"](in_channels=2, out_channels=2)
y = model(torch.randn(1, 2, 64, 64))    # -> torch.Size([1, 2, 64, 64])
```

Counting for yourself:

```python
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY
from spectramr.models.losses.registry import LossRegistry
from spectramr.core.metrics.registry import MetricsRegistry

populate_model_registry()
paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS   # a CLASS attribute
len(MODEL_REGISTRY), len(LossRegistry.list_available()), \
    len(MetricsRegistry.list_available()), len(set(paths.values())), len(paths)
```

The same model from a YAML:

```yaml
# an excerpt of experiments/templates/comprehensive_config_template.yaml
config_version: '1.0'
model:
  model_type: toeplitz_attention_unet
  in_channels: 2
  out_channels: 2
training:
  training_mode: reconstruction
  strategy_class: spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy
```

`config_version: '1.0'` is the only accepted value; anything else is refused at
load, with the accepted set named in the error.

The excerpt above is a fragment, not a runnable config. The complete template
ships and passes the audit as-is:

```bash
spectramr audit experiments/templates/comprehensive_config_template.yaml
spectramr train --config experiments/templates/comprehensive_config_template.yaml
```

`audit` runs Tier 0 (schema) and Tier 1 (health checks); add `--probe` for a
Tier 2 synthetic forward pass. It exits 0 on a pass, 1 on warnings, 2 on errors
-- and it is `--strict` by default, so a warning is not a pass.

## Installation

Optional dependencies come in two kinds. **Feature** groups gate a capability
(absent, it raises at construction — never a silent fallback); **role** groups
gate a workflow and are imported by nothing under `src/`.

```bash
pip install spectramr                 # core only
pip install spectramr[mri]            # TorchIO, MONAI, nibabel, torchkbnufft, pydicom
pip install spectramr[diffusion]      # diffusers — pretrained SD-VAE backbone
pip install spectramr[viz]            # matplotlib, tensorboard, seaborn, plotly
pip install spectramr[hpo]            # Optuna
pip install spectramr[all]            # EVERYTHING that installs in one shot
pip install spectramr[dev]            # all + the config-migration toolchain
```

The role groups are installable on their own, which is what CI lanes do:
`[test]` (pytest + plugins), `[qa]` (ruff, mypy, pre-commit, pip-audit,
codespell, detect-secrets), `[docs]` (Sphinx) and `[profile]` (Scalene, GPUtil,
nvtx). `[all]` contains all four, so it is a superset of whatever any lane
installs, and `[dev]` is `[all]` plus tooling nothing else needs.

Three groups are **not** in `[all]`, each because it physically cannot install
in a single resolve — not as a matter of curation, and each verified by an
actual build rather than assumed: `mamba` compiles the CUDA selective-scan
kernel; `attention` fails because flash-attn omits torch from its build
requirements; `radiomics` has no cp312 wheel and its C extension fails to
compile. (`bnb` and `deepspeed` were long excluded on the assumption that they
need a CUDA toolchain — both build clean under isolation, so they are in.) On a
node with `nvcc`:

```bash
pip install -e '.[all]'
make install-mamba                   # NOT `pip install -e '.[mamba]'` -- see below
```

The `mamba` extra goes through `scripts/install_mamba_extra.py` because the bare
pip line builds a kernel that cannot run on a V100 and reports success:
mamba-ssm and causal-conv1d download a prebuilt wheel unless `MAMBA_FORCE_BUILD`
/ `CAUSAL_CONV1D_FORCE_BUILD` are set, their hardcoded `-gencode` list omits
`sm_70` at every CUDA version, and `TORCH_CUDA_ARCH_LIST` is inert against a
package that passes its own `arch=` flags. The script forces the source build,
appends the gencode pairs through `NVCC_APPEND_FLAGS` (which nvcc itself reads),
defaults to `7.0;8.9` for the clusters' Volta and Ada nodes, and verifies the
result with `cuobjdump --list-elf` rather than trusting the flags.

`[mri]` is the practical floor, not a convenience. The core install is a
genuine subset and it is a small one: it registers **175** of the 586 models,
and `spectramr.infrastructure.training` cannot be imported at all, because the
dataset layer imports TorchIO unconditionally. `spectramr --help`,
`spectramr --version` and the loss registry (217, unaffected) still work. Install
`[mri]` unless you are deliberately vendoring a subset.

A few registered components need a package that no extra installs -- see
[docs/known_limitations.rst](docs/known_limitations.rst).

### Pinning the CUDA build

The `pytorch-cu126` pin in `pyproject.toml` lives under `[tool.uv.sources]` and
`[[tool.uv.index]]`, and **neither reaches wheel metadata** -- the published
requirement is a bare `torch>=2.8`. Installing from PyPI therefore resolves the
newest CUDA build rather than the pinned one: measured on a fresh venv from the
published wheel, `torch 2.13.0+cu130` plus the whole `nvidia-*-cu13` stack,
`cuda-toolkit` and `triton` -- **5.1 GB**, none of it asked for.

That matters on Volta. **cu126 is the last wheel lane that still ships `sm_70`**,
so a V100 (compute capability 7.0) fails every kernel launch on a cu13x build
with `cudaErrorNoKernelImageForDevice`. Install torch **first**, from the index
you want; doing it afterwards costs a multi-gigabyte reinstall:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # V100 / sm_70
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU-only, air-gapped
pip install spectramr[mri]
```

CI runs against the CPU wheel; downstream GPU work is yours to pin.

## Bring your own code

spectraMR is `pip`-installable, so a model, loss, metric, dataset or training
strategy of your own can live **outside** this repository and still be selected
by name from a YAML config or the scripting API. Nothing is forked and nothing is
subclassed: you register a component, and the framework imports it.

### 1. Register it

The decorator *is* the registration -- it runs on import:

```python
# my_pkg/models/my_unet.py
import torch.nn as nn
from spectramr import register_model

@register_model("my_unet", "reconstruction")
class MyUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, **kwargs):
        super().__init__()
        self.body = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        return self.body(x)
```

`register_loss` and `register_metric` are exported alongside it and work the same
way. Training strategies are the one exception: they resolve from a dotted-path
map rather than a decorator registry, so a plugin strategy is named either by its
full path in `training.strategy_class` or by a short name declared in the
`spectramr.strategies` entry-point group.

### 2. Make it discoverable

A decorator only fires if something imports the module. Three layers do that, and
they differ in exactly one way that matters -- what happens when the import fails:

| Layer | Declared in | On failure |
|---|---|---|
| Entry points | your package's `pyproject.toml` | **warns** -- a broken third-party plugin must not kill an unrelated run |
| `SPECTRAMR_PLUGINS` | the environment | **raises** |
| `plugins.paths` | the experiment YAML | **raises** |

The last two are things *you* declared, so an unimportable path raises
`PluginImportError` at startup rather than silently doing nothing, and the
resolved list is stamped into the run's `provenance.json`.

**Entry points** — for a shareable, installable plugin distribution. The five
groups are `spectramr.models`, `.losses`, `.metrics`, `.datasets` and
`.strategies`:

```toml
[project.entry-points."spectramr.models"]
my_unet = "my_pkg.models.my_unet"              # imported -> fires @register_model

[project.entry-points."spectramr.strategies"]
my_paradigm = "my_pkg.strategies.MyStrategy"   # short name -> dotted path
```

**Environment variable** — for a scratch script or a one-off override; paths are
separated by the OS path separator or by whitespace:

```bash
export SPECTRAMR_PLUGINS="my_pkg.models.my_unet my_pkg.losses.my_loss"
spectramr train --config experiment.yaml
```

**Config block** — for an experiment that should carry its own dependencies:

```yaml
plugins:
  enabled: true          # gates `paths` only; the other two layers run regardless
  paths:
    - my_pkg.models.my_unet

model:
  model_type: my_unet    # resolves -- the plugin was imported first
```

### 3. Names do not collide silently

The in-tree registry is populated **first**, so re-registering a name the
framework already owns is a hard error rather than an override: an explicit
`SPECTRAMR_PLUGINS` or `plugins.paths` collision raises, and an entry-point
collision warns while the in-tree component wins. A third-party package must
never quietly replace a framework one. Pick `my_unet`, not `unet`.

Full guide, including the in-process scripting API:
[docs/plugins.rst](docs/plugins.rst).

## What's in the box

| Layer | Where | Highlights |
|---|---|---|
| Training paradigms | `spectramr.infrastructure.training.strategies` | GAN, diffusion (cold / score / Lévy / resetting), VAE/VQ-VAE, MAE/SSL, reconstruction, domain adaptation, physics-driven (PINN), disentangled, sensitivity-estimation, cycle-Bloch |
| Model registry | `spectramr.models` | U-Nets, complex U-Nets, attention-bottleneck nets, geometric-prior nets (hyperbolic, Heisenberg, tropical, sheaf, …), state-space (S4D / Hyena), Toeplitz / Bloch-LRS / Lanczos / MPS attention |
| Loss registry | `spectramr.models.losses` | image, k-space, complex, physics-residual, adversarial, latent, distillation, virtual-fiducial, intertwining (spectral-triple) |
| Physics SSOT | `spectramr.infrastructure.physics` | `fft2c`/`ifft2c`, mask generators (Cartesian, VD, radial, NUFFT, SLE-κ), ESPIRiT, SENSE, PINN, Bloch, motion, B0, B1⁻ |
| Configuration | `spectramr.config` | `config_version: '1.0'` frozen Pydantic v2 schema, paradigm-specific sub-schemas, three-tier audit ladder |
| CLI | `spectramr.cli` | 24 verbs; `spectramr --help` lists them. The common ones are `audit`, `train`, `predict`, `infer`, `benchmark`, `hpo`, `report`, `doctor` |

## Maturity by regime

A **regime** is the physical acquisition setting an experiment declares
(`workflow.regime`). Each is graded against the live registries by
`spectramr.config.schemas.enums.Maturity`, and the grades are enforced by
`tests/unit/domain/workflows/test_maturity_ledger.py` -- they are read off the
code, not maintained by hand.

| Maturity | Regimes |
|---|---|
| **LIVE** -- registered forward model, regime-tagged strategy and metrics | `mri_structural`, `mri_quantitative`, `mri_diffusion_weighted`, `mri_dynamic`, `mri_functional`, `mri_perfusion`, `mri_flow`, `mri_spectroscopy`, `mri_fingerprinting` |
| **STUB** -- nothing exists; every pipeline raises `WorkflowNotImplementedError` | `ct`, `xray`, `ultrasound`, `optical`, `nmr_spectroscopy` |

The five STUB regimes are declared so the vocabulary is closed and a typo raises
instead of silently meaning nothing. They are not implemented, and spectraMR does
not claim to be a CT, X-ray, ultrasound or optical framework.

The ledger grades **regimes**, not individual models. No per-model guarantee is
made or implied.

## Citing

If you use spectraMR in your research, please cite the software:

```bibtex
@software{gdihi2026spectramr,
  author    = {Gdihi, Adnane},
  title     = {spectraMR: A Multi-Paradigm Research Framework for MRI Reconstruction, Super-Resolution, and Generative Modelling},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22291316},
  url       = {https://github.com/adnaneGdihi/spectramr},
  version   = {0.1.0}
}
```

GitHub renders a "Cite this repository" button from [CITATION.cff](CITATION.cff)
that produces this BibTeX automatically. Citation tools that consume CFF 1.2.0
will pick the version up directly.

## Documentation

Full documentation is hosted at https://spectramr.readthedocs.io and follows the
[Diátaxis](https://diataxis.fr) quadrants:

- **Tutorials** — guided walk-throughs from `pip install` to a first
  reconstruction.
- **How-to guides** — add a paradigm, add a model, add a loss, write an
  experiment YAML.
- **Reference** — auto-generated API documentation plus the YAML-schema
  reference and registry catalogues.
- **Explanation** — clean-architecture layering, the audit ladder, the physics
  SSOT discipline.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version:

1. Fork, branch off `main`, implement.
2. `pre-commit install` and let it run on every commit.
3. `pytest -m "not gpu"` must pass. If you touched a YAML config, run
   `spectramr audit <path>` on it -- the audit is `--strict` by default, so a
   warning is a failure.
4. PR title uses a Conventional Commits prefix; commits use `git commit -s`
   (DCO sign-off).
5. CI runs a single required aggregator over: changed-line lint, repository
   guards, architecture fitness functions, a collection pass over the unit
   suite, a physics check, a config-schema audit, and a security scan. The
   collection pass **imports** every unit-test module without executing the
   tests, so a green lane is not "the unit suite passed" -- run `pytest`
   locally.

By participating you agree to the
[Contributor Covenant 2.1](CODE_OF_CONDUCT.md).

## Licence

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution of
upstream dependencies.

## Disclaimer

spectraMR is **NOT FOR CLINICAL USE**. It is research software and has not been
evaluated by any regulatory authority. See [DISCLAIMER.md](DISCLAIMER.md).
