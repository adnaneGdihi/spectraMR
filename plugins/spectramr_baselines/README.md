# spectramr-baselines

Prior-method k-space reconstruction baselines for the `kspace_filling` cohort, packaged as a
spectramr plugin so that third-party code stays out of `src/spectramr/`.

| Model name | Paper | Upstream (submodule under `external/baselines/`) |
|---|---|---|
| `cdiffmr_baseline` | Huang et al., CDiffMR, MICCAI 2023 | `cdiffmr` — ayanglab/CDiffMR |
| `fdb_baseline` | Mirza et al., Fourier-Constrained Diffusion Bridges, arXiv:2308.01096 | `fdb` — icon-lab/FDB |
| `shen2024_baseline` | Shen et al., Sci. Reports 14 (2024) | `shen2024` — GuoyaoShen/K-SapceColdDIffusion |

Each adapter imports the authors' network and forward process from the submodule and
calls them unmodified; `UpstreamProcessStrategy` drives the adapter's own
`training_loss` and `validation_sample` rather than this repository's diffusion loop.

## Install

```bash
git submodule update --init external/baselines
pip install -e plugins/spectramr_baselines
```

The Shen 2024 upstream imports `fastmri` at module level. It uses only the FFT, coil and
data modules, so `pip install fastmri==0.3.0 --no-deps` is enough when you do not want
pytorch-lightning pulled in.

## Use

The arms under `experiments/inprogress/kspace_filling/baseline_*.yaml` name their adapter
in `plugins.paths` and their strategy by dotted `strategy_class`, so they load with the
plugin either installed or on `PYTHONPATH` (`plugins/spectramr_baselines/src`).

## FDB validation

The authors test at R = 4 on masks shipped inside their test `.mat` files and not
published, so `fdb_baseline` regenerates a 2-D normal-density point mask from the
paper's description (`test_acceleration`, `test_mask_sigma`, `test_mask_seed`, all in
`provenance()`). The reverse bridge then runs for `T_f (R - 1) R' / ((R' - 1) R)` steps
(1500 at R = 4 from a bridge trained at R' = 2), and a mask the schedule of
`create_mask` cannot consume is refused before the loop starts, since upstream would
spin on it forever.

## Licences

CDiffMR and FDB ship licence files. The Shen 2024 repository ships none, so it is
recorded as a submodule pointer only and nothing from it is redistributed here.
