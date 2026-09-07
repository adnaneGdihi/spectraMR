.. _getting_started:

===============
Getting Started
===============

spectraMR is driven from a YAML file and a CLI verb. This page takes you from an
empty environment to a trained model: install the package, write one
configuration, check it, train it, and read the results.

.. contents:: On this page
   :local:
   :depth: 2

System requirements
===================

Hardware
--------

============  ===================================  =====================================
Resource      Minimum                              Comfortable
============  ===================================  =====================================
CPU           4 cores                              8+ cores
RAM           16 GB                                32+ GB
Storage       50 GB free                           200+ GB SSD
GPU           NVIDIA, 8 GB VRAM                    NVIDIA, 16+ GB VRAM
============  ===================================  =====================================

A GPU is not optional for training. Every heavy pipeline — ``train``, ``infer``,
``hpo``, ``ablation``, and the ``audit --probe`` forward pass — runs on an
accelerator or raises rather than falling back to CPU. CPU is reachable only when
you ask for it explicitly (``--device cpu``, or ``device: cpu`` in the YAML). See
:doc:`accelerated_run_contract`.

Software
--------

- **Operating system**: Linux, or Windows with WSL2. macOS has no CUDA lane,
  so it can run the CLI and the CPU-only paths but not training.
- **Python**: 3.12 or newer.
- **CUDA**: 12.6. The project pins the ``pytorch-cu126`` wheel index
  deliberately — cu126 is the last lane that still ships ``sm_70`` kernels, which
  V100-class cards need. A cu129 or newer wheel fails every kernel launch on a
  V100 with ``cudaErrorNoKernelImageForDevice``.

Installation
============

There are two routes. Install from PyPI if you want to *run* the framework;
clone if you also want the example configurations and the source.

From PyPI
---------

.. code-block:: bash

   pip install spectraMR              # core
   pip install "spectraMR[mri]"       # + TorchIO / MONAI / NiBabel / torchkbnufft
   pip install "spectraMR[all]"       # everything that resolves in one shot

From source
-----------

.. code-block:: bash

   git clone https://github.com/adnaneGdihi/spectramr.git
   cd spectramr

   python3.12 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate

   pip install -e ".[all]"

``[all]`` is every feature group that resolves in a single ``pip install``. Three
extras are deliberately outside it because they cannot build under isolation:
``mamba``, ``attention`` and ``radiomics``.

Mamba and SSM models (``hilbert_mamba``, ``geomamba``, ``bloch_mamba``, …) need
the official CUDA selective-scan kernel, which compiles from source. Install it
as a second step on a machine with ``nvcc``:

.. code-block:: bash

   pip install -e ".[mamba]" --no-build-isolation

Check the installation
----------------------

Both routes install the ``spectramr`` console script. Every command on these
pages uses it.

.. code-block:: bash

   spectramr --help
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

From a clone you can also confirm that every dependency an extra declares is
installed, version-correct and importable:

.. code-block:: bash

   python scripts/verify/verify_dependencies.py --all --import-check

Your first configuration
========================

An experiment is one YAML file. Nothing else — there is no per-experiment Python
to write. Save the following as ``my_first_run.yaml``. It trains a complex-valued
U-Net to reconstruct 4×-undersampled Cartesian k-space from the M4Raw dataset,
and it completes in well under a minute on one GPU.

.. code-block:: yaml

   config_version: '1.0'

   run:
     device: cuda
     seed: 42

   workflow:
     regime: mri_structural
     task: reconstruction

   model:
     model_type: complex_unet
     in_channels: 2
     out_channels: 2
     spatial_dims: 2

   data:
     dataset_type: m4raw
     trajectory: cartesian
     coils:
       processing_mode: rss
     target_mode: phase_aligned_mean
     pairing:
       single_contrast: true
     datasets:
       - name: m4raw_multicoil
         path: databases/m4raw/data/multicoil_train
     split:
       type: auto
       validation_fraction: 0.15
     processing:
       data_range: 70.0
     sampling:
       patch_size: [256, 256, 1]
       samples_per_volume: 4
       queue_length: 32
     loader:
       batch_size: 1
       num_workers: 2

   undersampling:
     base_acceleration: 4.0
     center_fraction: 0.08
     acceleration_type: cartesian_vd

   training:
     task: reconstruction
     input_domain: kspace
     output_domain: kspace
     strategy_class: spectramr.infrastructure.training.strategies.reconstruction.ReconstructionTrainingStrategy
     epochs: 1
     output_dir: experiments/results/my_first_run

   optimization:
     optimizer:
       type: adam
       learning_rate: 0.0001

   validation:
     schedule:
       on_epoch: true
       interval_epochs: 1

   logging:
     identity:
       experiment: my_first_run
     intervals:
       log: 10
       save: 5

   losses:
     image_losses:
       - name: l1
         weight: 1.0
     policy:
       output_domain: kspace
     reconstruction:
       warmup_iterations: 0

   metrics:
     compute: [psnr, ssim]
     train_metric_interval: 6

   checkpoint:
     checkpoint_dir: experiments/results/my_first_run/checkpoints

What these keys do
------------------

``config_version: '1.0'``
   The configuration schema version. It is required, and ``'1.0'`` is the
   current one.

``run.device`` / ``run.seed``
   Where the run executes and what seeds it. ``run`` is the canonical home for
   both.

``workflow.regime`` / ``workflow.task``
   What kind of imaging problem this is. Both are closed vocabularies. Declaring
   them buys extra validation: the checker can then reject a component that does
   not belong in this regime.

``model.model_type``
   The registered model to build. ``complex_unet`` is registered as a *k-space*
   model, which is the domain this data route produces — so the model consumes
   the loader's output directly and no adapter chain is needed.
   ``in_channels``/``out_channels`` are ``2`` because one complex coil is carried
   as interleaved real and imaginary channels.

   Choosing a model whose registered domain matches your data is the difference
   between a configuration that runs and one that needs a bridge. The domain each
   model declares is in :doc:`model_capabilities`.

``data.datasets`` and ``data.split``
   Point at a directory of ``.h5`` files and let ``split.type: auto`` carve a
   validation fraction out of it. This is the simplest data route; a manifest
   file (``data.source.index_path``) is the alternative and is described under
   `Using a manifest instead of a directory`_.

``data.coils.processing_mode: rss``
   How the four receive coils are combined. ``rss`` reduces them to one complex
   coil, which is what makes ``in_channels: 2`` correct.

``data.pairing.single_contrast: true``
   Pair each volume with itself rather than with a second contrast. Without it
   the loader concatenates two contrasts along the channel axis and hands the
   model twice the channels it declares.

``data.sampling.patch_size``
   The patch the queue extracts. It must fit inside the volume: M4Raw is
   256×256, and the default ``[320, 320, 1]`` matches nothing, so every subject
   is filtered out and the run ends with an empty sampler.

``data.processing.data_range``
   The intensity range the range-sensitive metrics measure against. M4Raw
   magnitudes are not normalised to ``[0, 1]``, and ``psnr``/``ssim`` refuse to
   infer a range from unnormalised data — they report ``NaN`` with a
   ``NOT APPLICABLE`` warning rather than guessing. Declaring the range makes
   them compute.

``undersampling``
   The retrospective acceleration applied to fully sampled data.
   ``base_acceleration: 4.0`` keeps a quarter of the phase-encode lines;
   ``center_fraction: 0.08`` always keeps the central 8 %, which carries the
   low-frequency content.

``training.strategy_class``
   The training loop to run. Reconstruction is one strategy among many — see
   :doc:`strategies_reference`.

``training.output_dir``
   Where checkpoints, logs and reports land. It must begin with
   ``experiments/results/``; the health checker warns otherwise, and a warning
   fails the audit.

``validation.schedule``
   When validation runs. ``on_epoch: true`` with ``interval_epochs: 1``
   validates at the end of every epoch. A run that never validates has no
   ``val_*`` metrics and no best-metric checkpoint.

``losses.reconstruction.warmup_iterations: 0``
   **This line is load-bearing.** Several losses, ``l1`` among them, are ramped
   in over a warmup period that defaults to 1000 iterations. A configuration
   that declares only ``l1`` and leaves the default therefore has *zero
   gradient* for its first 1000 steps. Setting the warmup to ``0`` makes the
   loss live from step one.

``losses.policy.output_domain``
   Required whenever you configure losses as lists
   (``image_losses`` / ``kspace_losses`` / ``complex_losses``). It states the
   domain the losses are evaluated in.

``metrics.compute``
   The metrics to compute, by registered name. Names are validated against the
   registry, so a typo raises instead of silently computing nothing.

``metrics.train_metric_interval``
   How often training metrics are computed. It must be small enough to fire
   inside your iteration budget — the pipeline says so explicitly when it is
   not.

``checkpoint.checkpoint_dir``
   Keeps checkpoints inside the run directory. Without it they are written to
   ``./checkpoints``, outside the run, where an artifact bundle will not collect
   them.

Every key, with its default and its type, is in
:doc:`config_schema_reference`.

Check it before you run it
==========================

Run the audit before you spend GPU hours:

.. code-block:: bash

   spectramr audit my_first_run.yaml --probe

The audit loads the configuration, resolves every component you named, and runs
the full health-check suite over the result. With ``--probe`` it goes further: it
synthesises a batch, builds the model, and runs a real forward and backward pass
on the accelerator. **The probe needs no dataset** — it does not read one — so
this works before you have downloaded anything, and equally so it cannot tell
you whether your data route resolves. A green ``--probe`` says the model builds
and its gradients flow; only ``train`` says the loader agrees.

A clean run ends like this:

.. code-block:: text

   ✅ [strategy_class_matches_training_mode] strategy_class and training_mode agree
   ✅ [validation_metric_names_resolve] all 1 validation/selector name(s) resolve (strategy ReconstructionTrainingStrategy)
   ✅ [tier2_probe_accelerated] Tier-2 probe runs accelerated on cuda (source=run.device).
   ✅ [tier2_probe] complex_unet: forward (1, 2, 256, 256) -> (1, 2, 256, 256) + backward OK on cuda.

Read the exit code, not the volume of output:

- **0** — every check passed.
- **2** — at least one check failed *or* warned. Warnings are not tolerated:
  the audit runs strict, because a passed-with-warnings run is how dropped
  losses and validation-time OOM reach a cluster queue.

The audit also prints advisory lines marked ``📌`` and ``💡``. Those are
recommendations, not failures — they do not change the exit code.

Two failures are worth recognising on sight:

``domain_alignment``
   The channel count your model declares does not match what the adapter chain
   and coil handling actually produce. This one aborts the run before any GPU
   memory is allocated, which is the difference between a five-second failure
   and a five-hour one.

``the CONFIGURED loss g_total_loss is 0.0 at iteration 0``
   The probe computed your loss and got zero gradient. Usually the warmup trap
   described above.

``spectramr train --config my_first_run.yaml --dry-run`` is the lighter check:
it resolves the configuration and builds the services, then stops without
training. It does not run the probe. :doc:`audit_ladder_user_guide` describes
the audit's tiers in full.

Get some data
=============

The framework reads HDF5 and NIfTI volumes. Two public datasets are convenient
starting points.

M4Raw
-----

M4Raw is small, openly licensed (CC-BY-4.0), and needs no registration — the
quickest route to a real training run. Fetch it from
`its Zenodo record <https://doi.org/10.5281/zenodo.8056074>`_ and unpack it so
the multicoil training volumes sit at ``databases/m4raw/data/multicoil_train/``.
That is the path the configuration above already names.

FastMRI
-------

`fastMRI <https://fastmri.med.nyu.edu/>`_ requires an account and acceptance of
a data-usage agreement. Download the brain multicoil training set and unpack it.

Using a manifest instead of a directory
---------------------------------------

``data.datasets`` discovers the corpus from the directory. A *manifest* indexes
it once instead, recording each volume's path and shape, which makes the corpus
a fixed, reviewable list rather than whatever the directory holds today. From a
clone:

.. code-block:: bash

   python scripts/data/regenerate_cluster_manifests.py \
       --data-base databases \
       --datasets m4raw_multicoil_train

Pass ``--datasets`` one or more logical dataset names, or omit it to regenerate
every dataset found under ``--data-base``. Each writes a JSON file under
``data/manifests/``; ``--dry-run`` reports what would be written without writing
it. Point the configuration at the result by replacing the ``datasets``/``split``
block with:

.. code-block:: yaml

   data:
     source:
       root: databases/m4raw/data
       index_path: data/manifests/m4raw_train.json

Train
=====

.. code-block:: bash

   spectramr train --config my_first_run.yaml

The run finishes with a summary line, and everything it produced lands under
``training.output_dir``:

=========================  ==================================================
``checkpoints/``           ``checkpoint_epoch_<E>_step_<S>.pt``; a
                           ``checkpoint_best.pt`` appears once more than one
                           validation event has been scored
``logs/``                  ``training_metrics.csv`` and
                           ``validation_metrics.csv``
``resolved_config.json``   the fully resolved configuration, as run
``provenance.json``        run id, seed, device, config hash, git state
``final_metrics.json``     the metric values the run ended on
``debug_snapshots/``       input, prepared input and target for the first
                           steps, so you can see what the model was fed
``analysis/``              gradient logs
``report/``                written by ``spectramr report`` (below)
=========================  ==================================================

Each of ``resolved_config.json`` and ``provenance.json`` is also written with a
per-run suffix, so a directory reused across runs keeps every run's record
rather than overwriting it.

``resolved_config.json`` is the one to keep. It records every value the run
actually used, defaults included, so a result stays reproducible even after you
edit the YAML. :doc:`run_provenance_and_logging` describes what is stamped and
where.

Useful flags on ``train``:

``--resume auto``
   Continue from the latest checkpoint in the output directory.

``--override KEY=VALUE`` (``-O``)
   Change one value without editing the file, e.g.
   ``-O optimization.optimizer.learning_rate=0.0005``. Repeatable.

``--device cpu``
   The explicit opt-out from the accelerated-run contract.

Watch it train
==============

**Console.** Metrics print every ``logging.intervals.log`` iterations.

**TensorBoard.** Tracking is on by default
(``logging.tracking.service`` defaults to ``tensorboard``). The writer logs to a
``tensorboard/`` directory beside the run's other output — the run log names the
exact path as it starts:

.. code-block:: bash

   tensorboard --logdir experiments/results/my_first_run/tensorboard

Set ``logging.tracking.service: none`` to turn it off. Those two are the whole
vocabulary — there is no third tracking backend.

**CSV.** ``logs/training_metrics.csv`` accumulates every logged metric and is
what ``spectramr report`` reads.

If throughput rather than correctness is the problem,
:doc:`training_throughput` covers the knobs that move it.

Predict with the trained model
==============================

.. code-block:: bash

   spectramr infer \
       --checkpoint experiments/results/my_first_run/checkpoints/checkpoint_epoch_<E>_step_<S>.pt \
       --input <directory of input volumes> \
       --output output/my_first_run

``ls`` the ``checkpoints/`` directory and substitute the file you want.
A ``checkpoint_best.pt`` is written there instead once more than one
validation event has been scored.

``--config`` is optional here: the checkpoint's run directory carries
``resolved_config.json``, and ``infer`` reads it unless you pass ``--from-yaml``.
That is the reproducible path — it uses the settings the checkpoint was trained
with, not whatever the YAML says today.

``infer`` reads each input volume directly, so **the files you point it at must
already carry the channel layout the model expects.** It applies no coil
combination of its own: a checkpoint trained on RSS-combined data
(``in_channels: 2``) needs single-coil input, and a raw multi-coil volume is
rejected with a message naming both counts rather than being reshaped to fit.
Training-time coil handling belongs to ``data.coils.processing_mode``, and it
runs in the data pipeline, not here.

Report on the results
=====================

.. code-block:: bash

   spectramr report --exp-dir experiments/results/my_first_run

This reads the run's metric CSVs and writes into
``experiments/results/my_first_run/report/``: ``figures/`` (each figure as both
PNG and PDF, with a ``.meta.json`` beside it), ``report_summary.md``,
``qc_report.html`` and ``report_manifest.json``. It reports how many figures and
tables it produced, and says which figures it declined to draw and why — a
metric with no spread across cases is not plotted as a distribution.

Add ``--recursive`` to treat the directory as a cohort root and report on every
run beneath it. :doc:`reporting` covers the figure set and the ``reporting:``
configuration block.

Where to go next
================

**Learn the framework by example.** The :doc:`tutorials/index` build on this
page: :doc:`tutorials/tutorial_01_basic_reconstruction` takes the reconstruction
arm further, and :doc:`tutorials/tutorial_02_gan_super_resolution`,
:doc:`tutorials/tutorial_03_diffusion_training` and
:doc:`tutorials/tutorial_04_physics_constraints` each swap in a different
paradigm.

**Change what you configure.** :doc:`models_reference` and
:doc:`model_capabilities` list the models and what each one can do;
:doc:`strategies_reference`, :doc:`losses_reference` and
:doc:`metrics_reference` do the same for the other registries.
:doc:`config_schema_reference` is the exhaustive key reference.

**Run more than one experiment.** :doc:`hpo_guide` covers hyper-parameter
search, :doc:`execution_modes` and :doc:`running_pipelines` cover the other CLI
verbs, and :doc:`distributed_training` covers multi-GPU and multi-node runs.

**Extend it.** :doc:`tutorials/tutorial_05_custom_loss` registers a loss of your
own, and :doc:`plugins` loads components from outside the source tree.
:doc:`scripting_api` is the Python entry point for anything the CLI does not
cover.

**When something breaks.** :doc:`troubleshooting` is organised by the error
message you actually saw. :doc:`known_limitations` records what this release
does not do.
