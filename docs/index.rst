=========
spectraMR
=========

spectraMR is a multi-paradigm PyTorch research framework for MRI reconstruction,
super-resolution, contrast translation and synthesis. It is registry-dispatched
and driven from YAML: you select components by name in a configuration file and
run a CLI verb, and there is no per-experiment Python to write.

.. warning::

   **NOT FOR CLINICAL USE.** Research software only. It is not a medical device
   and has not been evaluated by any regulatory authority. See ``DISCLAIMER.md``.

.. note::

   **This is the public documentation set, and it is a subset.** It documents
   version |release|. These pages were written against the internal research
   tree, where an experiment corpus of several hundred configurations and a
   directory of internal tooling sit beside the framework. Neither is published.
   Pages here therefore sometimes *name* an internal script or an experiment arm
   as the provenance of a measured number. Naming a source is fine; telling you
   to **run** a file this repository does not contain is not, and any such block
   that survives here is a documentation defect worth an issue rather than an
   instruction worth following. :doc:`known_limitations` records what this tree
   does not do, and it is worth reading before the tutorials rather than after.

Start here
==========

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started
   tutorials/index
   how_to/index
   user_guide
   troubleshooting
   known_limitations

.. toctree::
   :maxdepth: 2
   :caption: Driving it from configuration

   config_schema_reference
   config_key_reference
   transform_registry
   environment_variables
   CLUSTER_DATA_LAYOUT

.. toctree::
   :maxdepth: 2
   :caption: Running the framework

   running_pipelines
   execution_modes
   cli_reference
   plugins
   campaigns_user_guide
   hpo_guide
   audit_ladder_user_guide
   accelerated_run_contract
   distributed_training
   training_throughput

.. toctree::
   :maxdepth: 2
   :caption: Results, logging and debugging

   run_provenance_and_logging
   reporting
   reporting_pipeline
   debug_snapshot_contract

.. toctree::
   :maxdepth: 2
   :caption: What you can configure

   models_reference
   model_capabilities
   model_registry_reference
   MODEL_TASK_MAPPING
   strategies_reference
   losses_reference
   metrics_reference

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index
   explanation/index
   versioning
   scripting_api
   sim2rank_reliability_theory
   modules/unet
   modules/loader
   modules/bloch

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
