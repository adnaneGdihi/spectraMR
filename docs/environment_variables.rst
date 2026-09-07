Environment variables
=====================

Every environment variable this framework reads or writes, how a value reaches
it, and how the launch backends differ in what they forward. For the commands
themselves see :doc:`cli_reference`.

Scope: what counts as an environment variable here
--------------------------------------------------

A name is documented here only when it is one of:

* read in Python via ``os.environ`` / ``os.getenv`` (including through a local
  ``FOO_ENV = "FOO"`` constant);
* expanded in shell with the ``${VAR:-default}`` / ``${VAR:?}`` idiom, which is the
  unambiguous "may come from the environment" form;
* a scheduler-, launcher- or tool-provided name (``SLURM_*``, ``CUDA_*``, ``PYTEST_*``);
* an overridable ``?=`` assignment in a :file:`Makefile`.

Shell variables that only look like knobs (``BLUE``, ``CONFIG_FILE``,
``AUDIT_EXIT_CODE``) are script-local plumbing and are not listed.

How a value reaches the framework
---------------------------------

Three surfaces are involved, and they are **not** equivalent:

:file:`.env.example`
   A copy-to-:file:`.env` template. :file:`.env` is gitignored and is sourced by
   the Makefile targets and the container entrypoint.

   Sourcing is a **shell** contract, not a Python one — nothing in the package
   calls ``load_dotenv()``. A launcher that does not source :file:`.env` never
   sees it, so a hand-rolled ``srun ... torchrun`` needs
   ``set -a; source "${REPO_ROOT}/.env"; set +a`` of its own.

:file:`src/spectramr/core/env.py`
   Name constants plus parsed-default helpers (``env.data_root()``,
   ``env.force_cpu()``, ``env.as_bool()``).

:file:`src/spectramr/infrastructure/config/env_resolver.py`
   Resolves the cache root and device preference at startup, and *writes*
   ``SPECTRAMR_CACHE_ROOT`` back into the environment so child processes inherit it.

Launch backends (``spectramr launch``)
--------------------------------------

``spectramr launch`` runs any pipeline anywhere: ``--where {local,docker,apptainer,slurm}``.
:doc:`execution_modes` describes the axes and the resource flags. What follows is
**how the environment reaches the run**, which differs per backend.

.. list-table::
   :header-rows: 1
   :widths: 14 30 56

   * - ``--where``
     - Environment inheritance
     - Notes
   * - ``local``
     - Full — it is your shell
     - Runs ``spectramr <verb> --config …`` in-process.
   * - ``slurm``
     - Full, via ``sbatch --export=ALL``
     - Emits an ``#SBATCH`` script from one ``ResourceSpec``. ``--account`` /
       ``--partition`` fall back to ``SPECTRAMR_SLURM_ACCOUNT`` / ``SPECTRAMR_SLURM_PARTITION``.
   * - ``apptainer``
     - Full — Apptainer inherits the host environment by default
     - Also passes ``--env CUDA_VISIBLE_DEVICES=…`` explicitly. Image from
       ``SPECTRAMR_APPTAINER_SIF`` (default ``spectramr.sif``).
   * - ``docker``
     - **``SPECTRAMR_LAUNCH_*`` only**
     - Docker inherits nothing. Set ``SPECTRAMR_DOCKER_IMAGE`` to an image you have
       built — no image is published for this framework. See the warning below.

.. warning::

   **A docker run does not see your** ``SPECTRAMR_*`` **configuration.** Only the
   ``SPECTRAMR_LAUNCH_*`` provenance variables are forwarded, so ``SPECTRAMR_DATA_ROOT``
   is unset inside the container and ``env.data_root()`` falls back to
   ``./databases`` — a *different data root* than the identical command run locally,
   with no warning. Pass what the run needs explicitly, or mount a data root the
   container can see.

``--dry-run`` prints the exact command or script without executing it, and is the
way to check what a backend actually forwards:

.. code-block:: console

   $ export SPECTRAMR_DATA_ROOT=/data/mine
   $ spectramr launch X.yaml --where docker --gpus 2 --dry-run
   docker run --gpus 2 --env SPECTRAMR_LAUNCH_BACKEND=docker ... \
     -v /repo:/workspace your-image:tag train --config X.yaml
   # note: SPECTRAMR_DATA_ROOT is absent

The ``SPECTRAMR_LAUNCH_*`` trio is *written* by ``launch`` onto the child run so its provenance records the backend and resources it ran under.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``SPECTRAMR_DOCKER_IMAGE``
     - *no published image*
     - Docker image for the ``docker`` execution backend; set it to an image you built.
       Read by ``src/spectramr/infrastructure/execution/backends.py``
   * - ``SPECTRAMR_APPTAINER_SIF``
     - ``spectramr.sif``
     - Apptainer ``.sif`` image for the ``apptainer`` backend.
       Read by ``src/spectramr/infrastructure/execution/backends.py``
   * - ``SPECTRAMR_SLURM_ACCOUNT``
     - *unset*
     - SLURM allocation account. No default ships: with neither this nor
       ``ResourceSpec(account=...)`` set, submission raises rather than sending
       ``--account=None`` to ``sbatch``.
       Read by ``src/spectramr/infrastructure/execution/backends.py``
   * - ``SPECTRAMR_SLURM_PARTITION``
     - *unset*
     - SLURM partition. Emitted as an ``#SBATCH`` directive only when set.
       Read by ``src/spectramr/infrastructure/execution/backends.py``
   * - ``SPECTRAMR_LAUNCH_BACKEND``
     - *unset*
     - Set \*by\* ``spectramr launch`` on the child run so provenance records the backend it ran under.
       Read by ``src/spectramr/infrastructure/execution/backends.py``
   * - ``SPECTRAMR_LAUNCH_ACCOUNT``
     - *unset*
     - Set \*by\* ``spectramr launch`` — resolved account, stamped into child provenance.
       Read by ``src/spectramr/infrastructure/execution/backends.py``
   * - ``SPECTRAMR_LAUNCH_GPUS``
     - *unset*
     - Set \*by\* ``spectramr launch`` — resolved GPU count, stamped into child provenance.
       Read by ``src/spectramr/infrastructure/execution/backends.py``

Distributed runs
----------------

``spectramr train-distributed`` is **not** launched directly — it expects a ``torchrun``
rendezvous, and ``torchrun`` is what sets ``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE`` /
``MASTER_ADDR`` / ``MASTER_PORT``. The distributed pipeline reads them back; see
`Distributed training`_ below for their defaults and readers.

.. note::

   ``launch --where slurm`` emits plain ``python -m spectramr.cli <verb>``, **not**
   ``torchrun`` — the unified launcher does not wrap distributed runs. For DDP,
   invoke ``torchrun`` directly.

:doc:`distributed_training` carries the launch recipes and the strategy settings.

Path resolution
---------------

Where the framework looks for data, caches and legacy mounts.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``SPECTRAMR_DATA_ROOT``
     - ``./databases``
     - Data root. Every ``${SPECTRAMR_DATA_ROOT}/...`` placeholder in YAML expands against it.
       Read by ``src/spectramr/data/metadata/path_resolver.py`` (+4 more)
   * - ``PROJECT_ROOT``
     - *unset*
     - Project root; synonym for the data root used by ``PathResolver`` / ``PathNormalizer``.
       Read by ``src/spectramr/data/metadata/path_resolver.py`` (+1 more)
   * - ``SPECTRAMR_ROOT``
     - *unset*
     - Third project-root spelling, read only by ``PathNormalizer`` and the config health checker.
       Read by ``src/spectramr/shared/utils/path_normalizer.py``
   * - ``SPECTRAMR_CLUSTER_ROOT``
     - the project root
     - Alternate cluster mount root used by ``cluster_explorer``.
       Read by ``src/spectramr/tools/cluster_explorer.py`` (+1 more)
   * - ``SPECTRAMR_CACHE_ROOT``
     - *unset*
     - Cache dir for weights and metric backbones. Also \*written\* by ``env_resolver``.
       Read by ``src/spectramr/infrastructure/config/env_resolver.py`` (+2 more)
   * - ``SPECTRAMR_LEGACY_ABS_PREFIXES``
     - *unset*
     - Colon-separated legacy absolute prefixes stripped before resolving.
       Read by ``src/spectramr/data/metadata/path_resolver.py`` (+1 more)
   * - ``SPECTRAMR_LEGACY_CLUSTER_PREFIX``
     - *unset*
     - Single legacy cluster prefix auto-rewritten to the project root.
       Read by ``src/spectramr/shared/utils/path_normalizer.py`` (+1 more)
   * - ``FASTMRI_DATASETS_ROOT``
     - *unset*
     - fastMRI-specific data-root override.
       Read by ``src/spectramr/shared/utils/path_normalizer.py`` (+1 more)

Device, determinism and the accelerated-run contract
----------------------------------------------------

Non-negotiable 9b: heavy pipelines run on an accelerator or raise. These are the only sanctioned ways to reach CPU.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``FORCE_CPU``
     - *unset*
     - The documented CPU escape hatch of the accelerated-run contract. Read \*and validated\* by ``compute_device`` — an unrecognized boolean raises.
       Read by ``src/spectramr/core/compute_device.py`` (+1 more)
   * - ``SPECTRAMR_DEVICE``
     - ``auto``
     - Device preference consumed by ``env_resolver`` (``auto`` by default).
       Read by ``src/spectramr/infrastructure/config/env_resolver.py`` (+1 more)
   * - ``SPECTRAMR_NO_GPU_PROBE``
     - *unset*
     - Skips the ``nvidia-smi`` probe in the container entrypoint. Read by shell, not Python.
       Read by ``src/spectramr/core/env.py``
   * - ``SPECTRAMR_GPU_MEMORY_FRACTION``
     - *unset*
     - Per-process GPU memory cap, float in (0, 1]. An invalid value raises at device init.
       Read by ``src/spectramr/infrastructure/training/utils/training_utils.py`` (+1 more)
   * - ``CUDA_VISIBLE_DEVICES``
     - *unset*
     - Standard CUDA device mask; read for provenance and for the CPU-fallback warning.
       Read by ``src/spectramr/infrastructure/training/utils/training_utils.py`` (+2 more)
   * - ``PYTHONHASHSEED``
     - ``0``
     - Hash seed; \*written\* by ``seed_control`` from the run's base seed.
       Read by ``src/spectramr/shared/utils/seed_control.py`` (+2 more)
   * - ``CUBLAS_WORKSPACE_CONFIG``
     - ``:4096:8``
     - cuBLAS determinism workspace; ``setdefault`` to ``:4096:8`` by ``pipelines/train``.
       Read by ``src/spectramr/core/env.py``

Framework behaviour and the CLI
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``SPECTRAMR_SUPPRESS_CLINICAL_WARNING``
     - ``1``
     - Silences the import-time clinical-use warning. Intended for batch jobs only.
       Read by ``src/spectramr/__init__.py`` (+2 more)
   * - ``SPECTRAMR_QUIET``
     - *unset*
     - Suppresses the heavy-startup notice for slow CLI commands.
       Read by ``src/spectramr/cli/app.py``
   * - ``SPECTRAMR_DEBUG``
     - *unset*
     - Forces a full traceback on CLI errors (same effect as ``-v``).
       Read by ``src/spectramr/cli/app.py``
   * - ``SPECTRAMR_PLUGINS``
     - *unset*
     - Extra plugin module paths to import, alongside the entry-point groups.
       Read by ``src/spectramr/plugins.py``
   * - ``SPECTRAMR_LEDGER_STRICT``
     - *unset*
     - Makes a missing execution-ledger stamp fatal. Off by default so a stamping hiccup never kills GPU work; on for CI and cluster audits.
       Read by ``src/spectramr/core/execution_ledger.py``
   * - ``SPECTRAMR_DIMENSION_CONTRACT``
     - ``observe``
     - Dimension-contract mode, one of the valid modes; defaults to ``observe``. An unrecognized value raises.
       Read by ``src/spectramr/infrastructure/validation/dimension_contract.py``
   * - ``SPECTRAMR_ALLOW_MAMBA_FALLBACK``
     - *unset*
     - Opts into the Gated-Conv+GRU fallback when ``mamba-ssm`` is absent. CPU/CI wiring tests only.
       Read by ``src/spectramr/models/blocks/mamba_block.py``
   * - ``FORCE_COLOR``
     - *unset*
     - Force ANSI colour in the logging service.
       Read by ``src/spectramr/infrastructure/services/logging_service.py`` (+2 more)
   * - ``NO_COLOR``
     - *unset*
     - Disable ANSI colour (the ``no-color.org`` convention).
       Read by ``src/spectramr/infrastructure/services/logging_service.py`` (+2 more)

Threading, caches and PyTorch tuning
------------------------------------

Most of these the framework **sets for you** — see `Variables the framework writes`_.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``OMP_NUM_THREADS``
     - ``1``
     - OpenMP threads. :file:`main.py` pins this at import, so exporting another value
       has no effect. Read by ``src/spectramr/main.py`` (+1 more)
   * - ``MKL_NUM_THREADS``
     - ``1``
     - MKL threads. Pinned at import the same way.
       Read by ``src/spectramr/main.py`` (+1 more)
   * - ``OPENBLAS_NUM_THREADS``
     - ``1``
     - OpenBLAS threads. Pinned at import the same way.
       Read by ``src/spectramr/main.py`` (+1 more)
   * - ``PYTORCH_CUDA_ALLOC_CONF``
     - *unset*
     - Allocator config; ``main.py`` writes ``expandable_segments:True,max_split_size_mb:512``.
       Read by ``src/spectramr/main.py`` (+4 more)
   * - ``CUDA_CACHE_MAXSIZE``
     - *unset*
     - CUDA JIT cache cap; written by ``main.py`` (2 GB).
       Read by ``src/spectramr/main.py`` (+1 more)
   * - ``CUDA_CACHE_CONFIG``
     - ``<cache root>/cuda_cache``
     - CUDA JIT cache dir; written under the cache root.
       Read by ``src/spectramr/accelerator.py`` (+3 more)
   * - ``TORCH_HOME``
     - ``<cache root>/torch_cache``
     - Torch hub cache; written under the resolved cache root.
       Read by ``src/spectramr/accelerator.py`` (+3 more)
   * - ``TORCH_METRICS_CACHE``
     - *unset*
     - torchmetrics cache dir; written by ``main.py``.
       Read by ``src/spectramr/main.py``
   * - ``TRITON_CACHE_DIR``
     - ``<cache root>/triton_cache``
     - Triton's cache dir, and the **one** member of the cache block applied with
       ``setdefault`` — see `The cache block has exactly two knobs`_.
       Read by ``src/spectramr/infrastructure/config/env_resolver.py`` (+1 more)
   * - ``TORCH_CUDA_EAGER_CACHE_MANAGER``
     - *unset*
     - Eager cache manager flag; written by ``main.py``.
       Read by ``src/spectramr/core/metrics/evaluation_metrics.py`` (+1 more)
   * - ``TORCHDYNAMO_VERBOSE``
     - ``0``
     - Dynamo verbosity; written, not read.
       Read by ``tests/conftest.py``
   * - ``TMPDIR``
     - ``/tmp/<user>``
     - Temp dir. Written in three places — see the \*hardcoded default\* warning below.
       Read by ``src/spectramr/infrastructure/config/env_resolver.py`` (+2 more)
   * - ``XDG_CACHE_HOME``
     - *unset*
     - XDG cache base; consulted by ``env.cache_root()`` and written by ``main.py``.
       Read by ``src/spectramr/main.py`` (+1 more)

The cache block has exactly two knobs
-------------------------------------

``configure_cache_environment`` (:file:`infrastructure/config/env_resolver.py`)
hangs six variables off one resolved root. Five are **assigned**; only
``TRITON_CACHE_DIR`` is ``setdefault``-ed. So of the six, exactly two respond to
anything you export:

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Yours to set?
     - Why
   * - ``SPECTRAMR_CACHE_ROOT``
     - **yes**
     - Read first when the root is resolved; moves all six at once.
   * - ``TRITON_CACHE_DIR``
     - **yes**
     - The one ``setdefault``. DeepSpeed's own startup warning asks operators to
       point Triton at a non-NFS path, so an explicit export is treated as an
       informed decision and kept.
   * - ``TMPDIR``, ``TORCH_HOME``, ``TORCH_METRICS_CACHE``, ``XDG_CACHE_HOME``,
       ``CUDA_CACHE_CONFIG``
     - no
     - Assigned from the root, overwriting whatever you exported. Deliberate: a
       site profile setting ``XDG_CACHE_HOME="$HOME/.cache"`` would otherwise put
       torch's JIT extension build root back inside ``$HOME``, which is the
       failure this block exists to prevent.

Every one of the six is created eagerly, so an unwritable path fails here rather
than inside ``import deepspeed`` (which writes to Triton's cache and to torch's
extension directory while the module is still executing). Since the failure is
also the operator's only instruction, it names the knob:

.. code-block:: text

   Cannot create the cache directory '/triton_cache' for TRITON_CACHE_DIR: Permission denied.
   TRITON_CACHE_DIR was INHERITED from the environment -- it is the one cache variable
   applied with setdefault rather than assignment (DeepSpeed asks operators to point
   Triton at a non-NFS path), so this value was NOT chosen by the framework.
   Either unset TRITON_CACHE_DIR to use the framework's own path
   (/tmp/$USER/spectramr_cache/triton_cache), or export it somewhere writable:
       export TRITON_CACHE_DIR="/tmp/$USER/triton_cache"
   A value whose first component is empty (e.g. '/triton_cache') is usually an unset
   ${PREFIX} in "${PREFIX}/triton_cache".

That is ``env_resolver.CacheDirectoryError``, an ``OSError`` subclass, so callers
already catching ``OSError`` keep catching it. The ``setdefault`` that makes ``TRITON_CACHE_DIR``
settable is the same property that makes an inherited *bad* value survive — the
framework cannot tell an informed NFS choice from an empty ``${PREFIX}``, so it
keeps the value and explains itself when the directory cannot be created.

.. tip::

   On a cluster, prefer node-local storage and include ``$USER``::

       export SPECTRAMR_CACHE_ROOT="/tmp/$USER/spectramr_cache"
       export TRITON_CACHE_DIR="/tmp/$USER/triton_cache"

   ``/tmp`` is sticky and shared on a compute node, so a bare ``/tmp/<name>``
   belongs to whichever user created it first. Keeping Triton's cache off network
   storage also avoids DeepSpeed's rank-exit hang.

Distributed training
--------------------

Set by ``torchrun`` or by the in-process launcher; read by the distributed pipeline.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``RANK``
     - ``os.environ.get("LOCAL_RANK", "-1``
     - Global rank. Read by the distributed pipeline; written by the in-process launcher.
       Read by ``src/spectramr/infrastructure/distributed/launcher.py`` (+2 more)
   * - ``LOCAL_RANK``
     - ``rank``
     - Node-local rank.
       Read by ``src/spectramr/infrastructure/distributed/distributed_training.py`` (+3 more)
   * - ``WORLD_SIZE``
     - ``-1``
     - Total process count; ``env.is_distributed()`` keys off it.
       Read by ``src/spectramr/infrastructure/distributed/launcher.py`` (+2 more)
   * - ``LOCAL_WORLD_SIZE``
     - ``world_size``
     - Node-local process count, read by ``pipelines/distributed``.
       Read by ``src/spectramr/pipelines/distributed.py``
   * - ``MASTER_ADDR``
     - ``127.0.0.1``
     - Rendezvous address; ``setdefault`` to ``127.0.0.1``.
       Read by ``src/spectramr/core/env.py``
   * - ``MASTER_PORT``
     - ``29500``
     - Rendezvous port; ``setdefault`` to ``29500``.
       Read by ``tests/unit/pipelines/test_train.py`` (+1 more)

Variables the framework writes
------------------------------

Some variables are set by the framework itself, which matters because a value
you export can be silently replaced. Two different disciplines are in use:

* ``os.environ.setdefault(...)`` — *yields* to a value you already set
  (:file:`accelerator.py`, :file:`infrastructure/distributed/launcher.py`,
  :file:`pipelines/train.py`).
* ``os.environ[...] = ...`` — **overwrites unconditionally**. :file:`main.py` does this
  at import time for the cache paths and for ``OMP_NUM_THREADS`` /
  ``MKL_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS``, which it pins to ``1``.

.. warning::

   Exporting ``OMP_NUM_THREADS=8`` before ``spectramr train`` has **no effect** —
   :file:`main.py` overwrites it with ``1`` on import. Thread count is an entry-point
   property here, not an environment one.

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Variable
     - Default
     - Read by

   * - ``CUBLAS_WORKSPACE_CONFIG``
     - ``:4096:8``
     - ``src/spectramr/core/env.py``
   * - ``CUDA_CACHE_CONFIG``
     - ``<cache root>/cuda_cache``
     - ``src/spectramr/accelerator.py``, ``src/spectramr/main.py`` (+2 more)
   * - ``CUDA_CACHE_MAXSIZE``
     - *unset*
     - ``src/spectramr/main.py``, ``src/spectramr/core/env.py``
   * - ``FASTMRI_DATASETS_ROOT``
     - *unset*
     - ``src/spectramr/shared/utils/path_normalizer.py``, ``src/spectramr/core/env.py``
   * - ``FORCE_COLOR``
     - *unset*
     - ``src/spectramr/infrastructure/services/logging_service.py``, ``tests/smoke/test_colored_logging.py`` (+1 more)
   * - ``SPECTRAMR_ALLOW_MAMBA_FALLBACK``
     - *unset*
     - ``src/spectramr/models/blocks/mamba_block.py``
   * - ``SPECTRAMR_CACHE_ROOT``
     - *unset*
     - ``src/spectramr/infrastructure/config/env_resolver.py``, ``tests/unit/infrastructure/config/test_env_resolver.py`` (+1 more)
   * - ``SPECTRAMR_LAUNCH_ACCOUNT``
     - *unset*
     - ``tests/unit/cli/test_launch.py``
   * - ``SPECTRAMR_LAUNCH_BACKEND``
     - *unset*
     - ``tests/unit/cli/test_launch.py``
   * - ``SPECTRAMR_LAUNCH_GPUS``
     - *unset*
     - ``tests/unit/cli/test_launch.py``
   * - ``SPECTRAMR_SUPPRESS_CLINICAL_WARNING``
     - ``1``
     - ``src/spectramr/__init__.py``, ``src/spectramr/cli/app.py`` (+1 more)
   * - ``LOCAL_RANK``
     - ``rank``
     - ``src/spectramr/infrastructure/distributed/distributed_training.py``, ``src/spectramr/infrastructure/distributed/launcher.py`` (+2 more)
   * - ``MASTER_ADDR``
     - ``127.0.0.1``
     - ``src/spectramr/core/env.py``
   * - ``MASTER_PORT``
     - ``29500``
     - ``tests/unit/pipelines/test_train.py``, ``src/spectramr/core/env.py``
   * - ``MKL_NUM_THREADS``
     - ``1``
     - ``src/spectramr/main.py`` (+2 more)
   * - ``OMP_NUM_THREADS``
     - ``1``
     - ``src/spectramr/main.py`` (+2 more)
   * - ``OPENBLAS_NUM_THREADS``
     - ``1``
     - ``src/spectramr/main.py`` (+1 more)
   * - ``PYTHONHASHSEED``
     - ``0``
     - ``src/spectramr/shared/utils/seed_control.py``, ``tests/unit/shared/utils/test_seed_control.py`` (+1 more)
   * - ``PYTORCH_CUDA_ALLOC_CONF``
     - *unset*
     - ``src/spectramr/main.py``, ``tests/integration/test_determinism_sentinel.py`` (+3 more)
   * - ``RANK``
     - ``os.environ.get("LOCAL_RANK", "-1``
     - ``src/spectramr/infrastructure/distributed/launcher.py``, ``src/spectramr/pipelines/distributed.py`` (+1 more)
   * - ``TMPDIR``
     - ``/tmp/<user>``
     - ``src/spectramr/infrastructure/config/env_resolver.py``, ``src/spectramr/main.py`` (+1 more)
   * - ``TORCHDYNAMO_VERBOSE``
     - ``0``
     - ``tests/conftest.py``
   * - ``TORCH_CUDA_EAGER_CACHE_MANAGER``
     - *unset*
     - ``src/spectramr/core/metrics/evaluation_metrics.py``, ``src/spectramr/main.py``
   * - ``TORCH_HOME``
     - ``<cache root>/torch_cache``
     - ``src/spectramr/accelerator.py``, ``src/spectramr/main.py`` (+2 more)
   * - ``TORCH_METRICS_CACHE``
     - *unset*
     - ``src/spectramr/main.py``
   * - ``TRITON_CACHE_DIR``
     - ``<cache root>/triton_cache``
     - ``src/spectramr/infrastructure/config/env_resolver.py``, ``src/spectramr/infrastructure/services/memory_optimization_service.py``
   * - ``WORLD_SIZE``
     - ``-1``
     - ``src/spectramr/infrastructure/distributed/launcher.py``, ``src/spectramr/pipelines/distributed.py`` (+1 more)
   * - ``XDG_CACHE_HOME``
     - *unset*
     - ``src/spectramr/main.py``, ``src/spectramr/core/env.py``
