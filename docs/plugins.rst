Out-of-Tree Plugins
===================

Because spectraMR is ``pip``-installable, you can define custom **models**,
**losses**, **metrics**, and **strategies** in your *own* package or a
standalone script — outside the ``spectramr`` source tree — and have them resolve
by name from a config or the :doc:`scripting_api`.

How registration works
-----------------------

A ``@register_model`` / ``@register_loss`` / ``@register_metric`` decorator is
*live* only once the module that defines it has been **imported** — importing is
what runs the decorator and adds the entry to the registry. The framework's own
components are imported by an in-tree walk; your out-of-tree components need one
of the three discovery layers below to be imported.

Strategies are different: they are resolved from a dotted-path map, not a
decorator registry. A plugin strategy is named either by a full dotted
``training.strategy_class`` (which already resolves with no extra wiring) or via
a short name declared in the ``spectramr.strategies`` entry-point group.

The three discovery layers
--------------------------

They run in this order at registry population; explicit, user-declared sources
fail loudly, while installed third-party plugins fail soft.

1. **Entry-points** — for shareable, pip-installed plugin distributions.
   Your package declares the components in *its own* ``pyproject.toml``:

   .. code-block:: toml

      [project.entry-points."spectramr.models"]
      my_unet = "my_pkg.models.my_unet"        # imported → fires @register_model

      [project.entry-points."spectramr.losses"]
      my_loss = "my_pkg.losses:MyLoss"

      [project.entry-points."spectramr.strategies"]
      my_paradigm = "my_pkg.strategies.MyStrategy"   # short-name → dotted path

   A broken third-party plugin is **warned about, not fatal** — it must not
   crash an unrelated training run.

2. **``SPECTRAMR_PLUGINS`` environment variable** — dotted module paths,
   separated by the OS path separator or whitespace. Best for a scratch script
   or a quick override:

   .. code-block:: bash

      export SPECTRAMR_PLUGINS="my_pkg.models.my_unet my_pkg.losses.my_loss"
      spectramr train --config experiment.yaml

   An unimportable token **raises** at startup (a user-declared knob must not
   silently no-op). The resolved list is stamped into the run's
   ``provenance.json``.

3. **``config.plugins`` block** — declare the import paths inside the config:

   .. code-block:: yaml

      plugins:
        enabled: true
        paths:
          - my_pkg.models.my_unet
          - my_pkg.losses.my_loss

      model:
        model_type: my_unet      # now resolves — the plugin was imported first

   Same raise-on-failure contract as the env var. These imports run *after* the
   in-tree registry walk but *before* the ``model_type`` check, so a config may
   name a plugin-provided model while a plugin cannot shadow an in-tree one.

Name collisions are an error
----------------------------

A plugin that re-registers a name an in-tree component already owns is a hard
error, not a silent override. The in-tree registry is
populated **first**, so the duplicate ``@register_*`` raises when the plugin is
imported:

* a ``SPECTRAMR_PLUGINS`` or ``config.plugins.paths`` collision **raises**
  (``PluginImportError``) — these are explicit, user-declared dependencies;
* an entry-point collision is **warned** about and the in-tree component wins —
  a third-party package must never silently replace a framework component.

Pick a unique name for your component (e.g. ``my_unet`` rather than ``unet``).

Minimal example
---------------

``my_pkg/models/my_unet.py`` (in your installed package or on ``PYTHONPATH``):

.. code-block:: python

   import torch.nn as nn
   from spectramr import register_model     # or: from spectramr.api import register_model

   @register_model("my_unet", "reconstruction")
   class MyUNet(nn.Module):
       def __init__(self, **kwargs):
           super().__init__()
           ...

Then either set ``SPECTRAMR_PLUGINS=my_pkg.models.my_unet``, add it to
``plugins.paths``, or declare the entry-point — and use ``model_type: my_unet``
in any config, or build it directly via :doc:`scripting_api`.

.. seealso::

   * :doc:`scripting_api` — the in-process scripting mode and public API.
   * :doc:`model_registry_reference` — the in-tree registries these layers feed.
