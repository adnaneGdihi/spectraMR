# Pre-1.0 checkpoint fixtures

Written by `CheckpointDirector` and `CheckpointService` as they stood at commit
`af740830060bb60df030de49f9cb1c578d1832f5`, the parent of the commit that added this directory. Both
writers were deleted by the checkpoint switch-over PR, and so was `_generate.py`.
To regenerate, check out that commit and run
`PYTHONPATH=src python tests/fixtures/checkpoints_pre_v1/_generate.py`.

- `director_run/`: two periodic saves (step 2 without early-stopping state, step 4
  with it), a best save, and the `best.pt` alias.
- `deepspeed_run/`: a ZeRO-2 fake engine. One periodic save, the consolidated
  `checkpoint_best.pt` (a bare state dict), the `checkpoint_best_ema.pt` sidecar,
  the alias, and the tag directories `epoch_0001_step_000004`, `best` and `latest`.
- `service_{pth,safetensors}_run/`: `CheckpointService` step and best files, the
  alias and `manifest.json`.
- `expected.json`: the epoch, step and first RNG draws after each file was restored
  by the pre-deletion `CheckpointDirector.load_from`.

`_generate.py` seeds torch, numpy and python to a different value before each save and
to an unrelated one before each restore, so `expected.json` only matches a resume that
really restored the saved streams. Text files (`expected.json`, `manifest.json`,
`latest`) gained a trailing newline from the end-of-file hook and must not be
byte-compared.
