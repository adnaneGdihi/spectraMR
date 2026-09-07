Config-declarable transform registry
====================================

``data.processing.transforms`` is the YAML seam for "run this transform on every
subject".

The registry
------------

:mod:`spectramr.data.transforms.registry` makes registry membership the validator,
exactly as ``MetricsRegistry`` + ``check_metric_names_are_registered`` does for
``metrics.compute``: a name that is not registered **raises**, and every
registered transform is reachable from YAML.

.. code-block:: python

   @register_transform("phase_residual", produces=("phase_residual",))
   class PhaseResidualTransform(tio.Transform): ...

.. code-block:: yaml

   data:
     processing:
       transforms:
         - name: phase_residual
           kwargs: {kernel_size: 9}
         - name: scout_acquisition
           kwargs: {scout_resolution: [32, 32]}

Kwargs may be nested under ``kwargs:`` (preferred) or written flat beside
``name`` — the flat spelling is what the committed ``graph_encoding`` arms use,
so it keeps working. An unknown kwarg is *not* swallowed: it reaches the
transform constructor and surfaces as a ``TypeError`` naming the transform.

``produces`` lets an audit answer "is there any registered producer for the key
this strategy reads?" without importing every strategy — see :func:`~spectramr.data.transforms.registry.transforms_producing`.

Where the transforms are applied
--------------------------------

:meth:`~spectramr.data.builders.torchio_transform_builder.TorchIOTransformBuilder._append_registry_transforms`
appends them to **both** the train and the val chain, last but one (before the
image→k-space bridge). Applying a declared transform to train only would
reproduce the normalization split this audit already found elsewhere: the model
would be graded on data the transform never touched.

Three layers of enforcement
---------------------------

#. **Schema** — ``TransformSpecSchema`` requires ``name``, which alone rejects
   the ``type:`` spelling at config load. Registry membership is deliberately
   *not* checked here: ``config/`` may not import ``data/``.
#. **Builder** — ``TorchIOTransformConfig.from_training_config`` resolves every
   name through the registry and raises ``KeyError`` listing the valid names,
   with an explicit hint when the name looks like a dotted import path.
#. **Audit** — ``check_transform_names_are_registered`` reports the same problem
   at Tier 1 (~100 ms), before anything loads data.

Adding a transform
------------------

#. Put the class in ``src/spectramr/data/transforms/`` — transforms have one
   canonical home.
#. Decorate it with ``@register_transform("<name>", produces=(...))``.
#. Import the module from ``src/spectramr/data/transforms/__init__.py`` — the
   decorator only runs when the module is imported, and a transform nobody
   imports is registered nowhere. This is the same maintenance contract as
   ``data/adapters/__init__.py``.
#. Add a test asserting it is registered and constructs with defaults.

Currently registered: ``foreground_mask``, ``graph_encoding``, ``phase_residual``,
``scout_acquisition``. The remaining caller-free transforms under
``data/transforms/`` are a separate decision — several are duplicates that
*disagree numerically* with a live implementation and should be deleted rather
than registered.
