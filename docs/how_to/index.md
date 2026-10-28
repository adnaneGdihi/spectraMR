# How-to guides

Task-focused recipes. Each one assumes you have the framework installed
(see [Getting started](../getting_started.rst)) and answers a single question of
the form *"how do I add X?"*.

If you are learning the framework rather than extending it, start with the
[tutorials](../tutorials/index.rst) instead — they build understanding in order,
where these guides assume you already know where things live.

| Guide | Use it when |
|---|---|
| [Add a training paradigm](add_paradigm.md) | You need a new `BaseTrainingStrategy` — a new way to train, not just a new network. |
| [Add a model](add_model.md) | You have a new generator/backbone to register in `models/registry.py`. |
| [Add a loss](add_loss.md) | You need a new objective term wired through `@register_loss` and the weight SSOT. |
| [Write an experiment YAML](write_experiment_yaml.md) | You are configuring an arm under `experiments/inprogress/<paradigm>/`. |

## The rules these guides encode

Every guide follows the same framework rules, which are worth knowing before you
start:

- **Register, don't branch.** Components resolve through registries and the DI
  container, never an `if/elif` chain — see {doc}`index`.
- **No silent fallbacks.** An unknown enum value must raise, never degrade to a
  default.
- **Every knob is wired.** If you expose a YAML key, something must read it,
  validate it, and stamp it into provenance in the same change.
- **Tests and docs land with the change.** Source and test files pair up in the
  same commit; `scripts/ci/check_test_paired_with_source.py` enforces it.

For the reasoning behind each, see {doc}`../config_schema_reference`.

```{toctree}
:maxdepth: 1
:hidden:

add_paradigm
add_model
add_loss
write_experiment_yaml
```
