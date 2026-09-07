=========
spectraMR
=========

spectraMR is a multi-paradigm PyTorch framework for MRI reconstruction,
super-resolution, contrast translation and synthesis. It is registry-dispatched
and driven from YAML: you select components by name in a configuration file and
run a CLI verb, and there is no per-experiment Python to write.

.. warning::

   **NOT FOR CLINICAL USE.** Research software only. It is not a medical device
   and has not been evaluated by any regulatory authority. See ``DISCLAIMER.md``.

:doc:`getting_started` installs the package and runs the first reconstruction.
From there the sections below follow the order you will need them in: describe a
run in a configuration file, launch it, choose the components it uses, and read
what it produced.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started
   tutorials/index
   troubleshooting
   known_limitations

.. toctree::
   :maxdepth: 2
   :caption: Configuring a run

   config_schema_reference
   explanation/workflows
   transform_registry
   environment_variables

.. toctree::
   :maxdepth: 2
   :caption: Running the framework

   cli_reference
   running_pipelines
   execution_modes
   hpo_guide
   audit_ladder_user_guide
   accelerated_run_contract
   distributed_training
   training_throughput

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
   :caption: Results and extension

   run_provenance_and_logging
   reporting
   how_to/index
   plugins
   scripting_api
   versioning
   modules/unet
   modules/loader
   modules/bloch

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
