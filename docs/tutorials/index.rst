.. _tutorials:

=========
Tutorials
=========

Guided walk-throughs that build understanding in order, from a first
reconstruction to custom losses and hyper-parameter search. Follow them with the
framework installed; they teach the concepts rather than assuming them. Once you
know your way around, the :doc:`../how_to/index` answer single questions instead.

.. contents:: On this page
   :local:
   :depth: 1

Core tutorials
==============

A sequence — each builds on concepts from the previous one. Start at Tutorial 1
with the framework installed; :doc:`../getting_started` covers the install and
the first run.

.. toctree::
   :maxdepth: 1

   tutorial_01_basic_reconstruction
   tutorial_02_gan_super_resolution
   tutorial_03_diffusion_training
   tutorial_04_physics_constraints

Extending the framework
=======================

These cross into how-to territory: they teach by adding a real component.

.. toctree::
   :maxdepth: 1

   tutorial_05_custom_loss
   tutorial_06_hyperparameter_optimization

Quick reference
===============

.. list-table::
   :header-rows: 1
   :widths: 5 25 20 15 15 20

   * - #
     - Tutorial
     - Difficulty
     - Time
     - GPU Req
     - What You'll Learn
   * - 1
     - :doc:`tutorial_01_basic_reconstruction`
     - Beginner
     - 2 hours
     - 8GB+ VRAM
     - U-Net training, evaluation, visualization
   * - 2
     - :doc:`tutorial_02_gan_super_resolution`
     - Intermediate
     - 4 hours
     - 12GB+ VRAM
     - Adversarial training, perceptual losses
   * - 3
     - :doc:`tutorial_03_diffusion_training`
     - Intermediate
     - 6 hours
     - 16GB+ VRAM
     - k-space Cold Diffusion, curriculum, importance sampling
   * - 4
     - :doc:`tutorial_04_physics_constraints`
     - Intermediate
     - 3 hours
     - 12GB+ VRAM
     - Data consistency, k-space constraints, Cycle-Bloch
   * - 5
     - :doc:`tutorial_05_custom_loss`
     - Advanced
     - 2 hours
     - 8GB+ VRAM
     - Custom loss functions, registry pattern, HFEN evaluation
   * - 6
     - :doc:`tutorial_06_hyperparameter_optimization`
     - Advanced
     - 3 hours
     - 8GB+ VRAM
     - Optuna HPO, search spaces, pruning, multi-objective

Prerequisites
=============

Before starting any tutorial:

- Work through :doc:`../getting_started`.
- Have at least one dataset ready (FastMRI or M4Raw).
- Have a GPU with sufficient VRAM (see the table above). Heavy pipelines raise
  rather than fall back to CPU; see :doc:`../accelerated_run_contract`.

Suggested paths
===============

**Learning the framework:** Tutorial 1 → 4 (physics constraints) → 2 (GAN).

**Research:** Tutorial 1 → 2 (GAN) → 3 (diffusion), then
:doc:`../execution_modes` for running many arms at scale.

**Extending:** Tutorial 1 → 5 (custom loss) → 6 (HPO), then
:doc:`../how_to/add_model` and :doc:`../how_to/add_paradigm`.

Where to go next
================

- :doc:`../how_to/index` — task-focused recipes.
- :doc:`../config_schema_reference` — every YAML key, with defaults.
- :doc:`../troubleshooting` — when a run misbehaves.
- :doc:`../cli_reference` — every command and flag.
