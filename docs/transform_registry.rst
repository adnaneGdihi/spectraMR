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

Currently registered (re-measure with ``TRANSFORM_REGISTRY``, this list moves):
``complementary_rep_masking``, ``espirit_sensitivity``, ``foreground_mask``,
``graph_encoding``, ``homomorphic_bias_field``, ``joint_rotation``,
``phase_residual``, ``scout_acquisition``, ``select_middle_slice``,
``simulate_ulf_from_hf``, ``simulate_ulf_from_qmaps``, ``slab_to_channel``,
``slice_profile``, ``svd_coil_compression``.

``svd_coil_compression`` (``requires=("input", "mask")``) is the M4Raw
repetition route's coil compression, which the loader's
``coils.processing_mode: svd`` cannot be on that route. Declared after
``complementary_rep_masking``, it fits one SVD basis per slice on the masked
``input`` alone and applies it to ``input``, ``measured_kspace``, ``target``,
``kspace`` and ``input_alt``. Compression is linear on the coil axis and the
mask acts on k-space positions, so the two commute and no unacquired line or
reference repetition chooses the basis. It raises when ``mask`` is absent or a
physical-coil key (coil maps, ``noise_covariance``) is present.

``espirit_sensitivity`` estimates complex ESPIRiT maps from the subject's
k-space and writes ``sensitivity_complex`` and its magnitude ``sensitivity``.
Under ``data.coil_maps_root`` the M4Raw loader has already attached the group's
precomputed maps and stamped the calibration that made them; the transform
then estimates nothing, and raises if that calibration or the maps' shape
differs from its own ``acs_size``/``kernel_size``/``eigen_threshold``.

``complementary_rep_masking`` (``produces=("mask", "loss_mask", "measured_kspace")``) is the
CoNEX protocol's data side (``experiments/inprogress/n2n/PAPER.md`` § 4): it
undersamples ``input`` by a mask :math:`M_1` drawn inside the acquired
phase-encode columns and emits ``loss_mask`` = :math:`M_2` for the
``masked_kspace_l2`` loss (:doc:`losses_reference`). ``mode`` selects how
:math:`M_2` relates to :math:`M_1` -- ``independent`` (a fresh draw),
``complementary`` (disjoint, subsampled to :math:`|M_2| = |M_1|` above R = 2),
``same`` and ``full_target`` (the two controls) -- and an unknown mode raises.
Seeds are a CRC32 of ``(file_id, slice, seed)`` so every arm of an ablation sees
the same :math:`M_1`. It also overwrites ``measured_kspace`` with the masked
input. The M4Raw dataset publishes that key as the unmasked ``input`` and serves
the reference under ``kspace``, so the strategy's measurement :math:`y` is the
acquired input on every M4Raw arm and never the target (#2761).

``pattern`` chooses the family of :math:`M_1`. ``cartesian_1d`` (the default) is the
uniform random line draw above. ``poisson_disk``, ``radial``/``golden_angle`` and
``spiral`` -- names resolved through ``SamplingPatternRegistry``, any other name
raising -- draw 2-D masks over the whole grid (``data/transforms/rep_mask_patterns.py``):
a variable-density Poisson-disc from ``MaskGenerator.variable_density_poisson_disc``,
or a seeded rotation of the golden-angle orderings rasterised onto the grid. R
means :math:`|M_1 \cap \mathrm{support}| = \mathrm{round}(|\mathrm{support}|/R)` for
every family, and every 2-D mask holds the centred
``round(center_fraction * W)``-square ESPIRiT window in both masks. A calibration
region that fills the budget raises instead of silently lowering R, and the achieved
R is stamped as ``realised_acceleration``. The remaining caller-free transforms under
``data/transforms/`` are a separate decision — several are duplicates that
*disagree numerically* with a live implementation and should be deleted rather
than registered.
