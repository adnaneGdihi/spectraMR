Scripting API (``spectramr.api``)
=================================

spectraMR supports four ways to launch work. Three are *declarative* — you write a
YAML config and the framework builds and runs everything:

* a single config-driven run (``spectramr train --config X.yaml``),
* a **campaign** that fans one research question out over many arms,
* a **container / cluster** backend (Docker, Apptainer, SLURM) that wraps a run.

The fourth is *imperative* — the **scripting mode** documented here. Because the
framework is ``pip``-installable, you can write a plain Python script that builds
the components you want and runs them directly, bypassing the YAML schema and the
audit ladder. This is the fast path for prototyping, and the way to use custom
losses / models / metrics that live **outside** the ``spectramr`` source tree.

.. note::

   Everything below is regular Python — you keep all of Python's composability
   (subclassing, closures, partials). The trade-off is that you also give up the
   guard rails the config path provides (schema validation, the ``audit``
   pre-flight). Use a config when you want those checks; use a script when you
   want speed and flexibility.

The public import surface
-------------------------

The public names are re-exported *lazily* from both the top-level package and the
:mod:`spectramr.api` facade (so ``import spectramr`` stays cheap and does not drag in
the whole torch import chain). Both share one export table, so they resolve to the
*identical* objects:

.. code-block:: python

   # either of these works and gives identical objects
   from spectramr import register_model, settings_from_dict, make_model
   from spectramr.api import register_model, settings_from_dict, make_model

Currently exported:

======================================  ==========================================
Name                                    Purpose
======================================  ==========================================
``fit`` / ``Trainer``                   Train a hand-built model in-process,
                                        reusing the standard loop.
``register_model``                      Decorator — register a model class by name.
``register_loss``                       Decorator — register a loss by name.
``register_metric``                     Decorator — register a metric by name.
``settings_from_dict``                  Build a validated, frozen
                                        ``TrainingSettings`` from an in-memory
                                        dict (no YAML file).
``TrainingSettings``                    The frozen config SSOT.
``make_model`` / ``make_optimizer``     Build a model or optimizer from a config
                                        (in-memory **or** a YAML path).
``make_dataset`` / ``make_dataloader``  Build a dataset or dataloader from a
                                        config (in-memory **or** a YAML path).
======================================  ==========================================

Training in-process: ``fit`` / ``Trainer``
------------------------------------------

``fit`` brings your own ``model`` / dataloaders and runs the **same** training
loop the config path uses — there is one loop engine, not two. The ``paradigm``
argument selects the training paradigm (the strategy + its sensible default loss
block); losses are built by the canonical ``LossBuilder`` so they are genuinely
consumed by the strategy, not a facade.

.. code-block:: python

   from spectramr.api import fit

   # reconstruction (default)
   result = fit(model, train_loader, val_loader=val_loader, device="cuda", epochs=50)

   # other paradigms — fit wires each one's components + default loss block:
   fit(model, train_loader, paradigm="diffusion")              # + training.diffusion
   fit(model, train_loader, paradigm="vae")                    # + losses.latent (KL)
   fit(g, train_loader, paradigm="gan", discriminator=d)       # + opt_d + losses.gan
   fit(model, train_loader, paradigm="physics_equivariant_ssl")

   # overrides: custom optimizer / losses / a ready strategy:
   #   fit(model, train_loader, optimizer=opt, losses={"l1": my_loss})
   #   fit(model, train_loader, strategy=my_strategy)   # used verbatim, any paradigm

``paradigm`` accepts any registered strategy short-name (an unknown value
raises), but only a curated set ship **tailored zero-config defaults** so
``fit(paradigm=...)`` just works:

* canonical families: ``reconstruction`` · ``diffusion`` · ``vae`` · ``vqvae`` ·
  ``gan`` · ``ssl`` · ``mae`` · ``physics_equivariant_ssl``.
* diffusion family (share the ``diffusion`` default — verified identical config
  contract): ``flow_matching`` · ``rectified_flow`` · ``fisher_rao_flow`` ·
  ``levy_diffusion`` · ``resetting_diffusion`` · ``tissue_diffusion_pretrain`` ·
  ``flow_matching_pfode`` · ``stochastic_interpolants`` · ``x_diffusion`` ·
  ``cross_modal_diffusion`` · ``edm`` (EDM/Karras).
* other verified: ``masked`` (MAE) · ``cartoon_texture_safe`` ·
  ``recoverability_vib`` · ``generative`` · ``universal_reconstruction``.

Reconstruction *variants* (any paradigm whose strategy is
``ReconstructionTrainingStrategy`` — e.g. ``dual_task``, ``hybrid``,
``multi_task``) also work with no extra config; ``fit`` detects them
automatically. Every *other* registered strategy is still runnable through
``fit`` — but you must pass a full ``config=`` (a ``TrainingSettings`` or a dict
with its training/losses blocks) or a ready ``strategy=``. Calling
``fit(paradigm=X)`` for such a paradigm **without** a config **raises** with
guidance rather than silently running with wrong defaults: strategies that need
physics/field data, an acquisition co-design loop, conformal post-hoc
calibration, federated multi-site streams, or MRF dictionaries are *deliberately*
not given fit defaults (a synthetic default would be a facade).

For non-recon paradigms the ``model`` you pass must be paradigm-appropriate (a
VAE model for ``vae``, etc.), and **GAN requires a ``discriminator=``** (or a
``model.discriminator_component`` in ``config``) — a GAN without one would
silently collapse to plain L1, so ``fit`` raises instead.

``Trainer`` is the reusable form — hold shared options once, call ``fit`` many
times:

.. code-block:: python

   from spectramr.api import Trainer

   trainer = Trainer(device="cuda", epochs=50)
   trainer.fit(model, train_loader, val_loader=val_loader)

Your dataloaders must yield batches as a mapping with the canonical keys
``"input"`` and ``"target"`` — the same contract the config-driven datasets
satisfy (``BatchAdapter`` raises on non-canonical keys). ``fit`` accepts a
partial ``config=`` dict (completed with the reconstruction defaults) or a full
``TrainingSettings`` for total control.

.. note::

   ``fit`` / ``Trainer.fit`` run the **full** training pipeline, so — like a
   config-driven run — they write a complete run bundle (checkpoints,
   ``*_metrics.csv``, provenance, an end-of-run HTML report) under the run's
   ``training.output_dir``. That default is **working-directory-relative**
   (it is *not* a temp dir), so a script run from your project root drops a
   bundle there. Redirect it explicitly when scripting:

   .. code-block:: python

      fit(model, train_loader, config={"training": {"output_dir": "/tmp/my_run"}})

Evaluating and predicting: ``Trainer.evaluate`` / ``Trainer.predict``
---------------------------------------------------------------------

``Trainer.evaluate`` runs ONE validation pass over a loader and returns the
aggregated metric dict — **no training, no optimizer steps**. It builds the same
env + strategy ``fit`` would and drives the *same* ``_run_validation`` the
training loop calls at each ``eval_interval`` (eval mode + EMA-swap), so the
numbers match in-training validation:

.. code-block:: python

   trainer = Trainer(paradigm="reconstruction", device="cuda")
   metrics = trainer.evaluate(model, val_loader)   # {"val_psnr": ..., "val_ssim": ...}

Unlike ``fit``, ``evaluate`` is cheap and writes no run bundle. It builds a
generator-only env (validation is generator-centric); for a paradigm whose
``validation_step`` needs more than the generator, pass a pre-built
``strategy=`` to the ``Trainer``.

``Trainer.predict`` is **path-based** — it delegates to the SSOT
:func:`~spectramr.pipelines.infer.run_inference_pipeline`, which is config +
checkpoint driven by design (data loading and result writes live in the data
layer via ``OutputWriter``, never in the scripting layer). It reconstructs the
run from its training YAML + a saved checkpoint, so it takes paths, not a live
model object:

.. code-block:: python

   trainer.predict(
       config_path="experiments/<paradigm>/<your-arm>.yaml",
       checkpoint_path="runs/my_arm/checkpoints/best.pt",
       input_path="data/test/",
       output_path="runs/my_arm/predictions/",
   )

This asymmetry (``evaluate`` in-memory, ``predict`` path-based) is intentional:
validation runs on the live model you just trained, whereas inference is a
config-and-checkpoint batch process owned by the data layer.

Registering an out-of-tree component
------------------------------------

Define your component anywhere — in your own package or a standalone script —
and decorate it. The decorator registers it by name immediately on import:

.. code-block:: python

   import torch.nn as nn
   from spectramr.api import register_model

   @register_model("my_unet", "reconstruction")
   class MyUNet(nn.Module):
       ...

``training_mode`` is required, not optional — it is the paradigm the model
belongs to (``"reconstruction"``, ``"gan"``, ``"diffusion"``, …), and omitting
it raises ``TypeError``.

For the component to resolve by name when the framework populates its registries
(e.g. on a config-driven run), the module that defines it must be *imported*. In
a script that you run yourself, importing it before you build the config is
enough. To have a pip-installed package or a YAML-named module discovered
automatically, see :doc:`plugins`.

Building components from an in-memory config
--------------------------------------------

``settings_from_dict`` is the in-memory peer of ``TrainingSettings.from_yaml``:
it validates and freezes a config built in code, applying the same dict-level
transforms (model-type registry check, coil-processing bridges) — so an
in-memory config behaves exactly like one loaded from YAML. ``config_version`` is
optional on this path.

.. code-block:: python

   from spectramr.api import settings_from_dict, make_model, make_optimizer

   cfg = settings_from_dict(
       {
           "model": {"model_type": "my_unet"},
           "data": {},
           "optimization": {"learning_rate": 1e-4},
           "logging": {},
       }
   )

   model, _ = make_model(config=cfg, device="cpu")
   optimizer, _ = make_optimizer(config=cfg, model=model)

The ``make_*`` helpers accept either an in-memory ``config=`` **or** a
``config_path=`` to a YAML file (the original form); passing neither raises.

Data helpers: ``make_dataset`` / ``make_dataloader``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both build through the canonical ``DataPipelineDirector``, so a scripted loader
is the same object training gets.

.. code-block:: python

   from spectramr.api import make_dataloader

   loader, meta = make_dataloader(config=cfg, split="train", device="cuda")
   meta["split_resolved"]   # the loader actually read — see the note below
   meta["shuffle"]          # read off the constructed loader, not inferred

``split`` accepts ``"train"``, ``"val"`` (or ``"validation"``) and ``"test"``.
**Anything else raises** rather than falling through to the *training* loader,
so a typo can never return training data under a bogus split name.

.. warning::

   There is **no held-out test set**: ``data.test_split`` is accepted by the
   schema and read by nothing. ``split="test"`` therefore returns the
   *validation* loader, and the metadata records that as
   ``split_resolved: "val"``. Do not report a number from it as a test-set
   result.

Batch size is **not** a parameter of these helpers — set
``data.loader.batch_size`` in the config. A helper-level ``batch_size=`` would
not be harmless: the director exposes only a validation-side batch-size knob,
the validation
*stride* is derived from it (so it changes which records the validation set
contains, not merely their grouping), and ``dataset_type: cine`` has a
batch-size guard it would route around.

.. seealso::

   * :doc:`plugins` — automatic discovery of out-of-tree components via
     entry-points and the ``SPECTRAMR_PLUGINS`` environment variable.
   * :doc:`cli_reference` — the config-driven and campaign launch modes.
