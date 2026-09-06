Runtime reference: environment, launch modes and commands
=========================================================

Everything you set or type to make this framework run, in one place: every
environment variable it reads or writes, the four launch backends and how the
environment reaches each one, the distributed-run invocations, and the full
command index.

Everything here is **measured** — the variables from the tracked tree, the command
table from the live ``argparse`` tree, the backend behaviour from ``--dry-run`` —
rather than transcribed from the partial lists that already exist.

Command index
-------------

**23 verbs**, read from the live ``argparse`` tree rather than transcribed.
``--config?`` is measured per subparser: a blank cell means the verb takes its
input another way (``audit`` takes the YAML as a positional). ``launch?`` marks the
**6** verbs reachable through ``spectramr launch --pipeline``.

See :doc:`cli_reference` for the full flag set of each verb; this table is the index,
not a replacement.

.. list-table::
   :header-rows: 1
   :widths: 20 8 8 64

   * - Command
     - ``--config``
     - ``launch``
     - Purpose
   * - ``spectramr doctor``
     - yes
     - —
     - Print environment diagnostics (torch/CUDA, devices, cache/data roots, env knobs).
   * - ``spectramr train``
     - yes
     - yes
     - Train a model.
   * - ``spectramr sanity_check``
     - yes
     - yes
     - Sanity check — overfit a single batch.
   * - ``spectramr ablation``
     - yes
     - yes
     - Train the baseline plus one variant per ``--vary`` override; report per-metric deltas. Sequential/local.
   * - ``spectramr infer``
     - yes
     - yes
     - Run inference using a trained model.
   * - ``spectramr infer-dataset``
     - yes
     - —
     - **Deprecated** — alias for ``infer``.
   * - ``spectramr experiment``
     - yes
     - yes
     - Run a complete experiment with directory management.
   * - ``spectramr train-distributed``
     - yes
     - —
     - DDP training. Not launched directly — see `Distributed runs`_.
   * - ``spectramr predict``
     - yes
     - —
     - Inference via the SSOT pipeline.
   * - ``spectramr benchmark``
     - —
     - —
     - Run benchmarks.
   * - ``spectramr export``
     - yes
     - —
     - Export a model.
   * - ``spectramr list-features``
     - —
     - —
     - List available models, losses, metrics, strategies.
   * - ``spectramr audit``
     - —
     - —
     - Audit an experiment YAML: Tier 0 schema, Tier 1 health checker, Tier 2 probe with ``--probe``. Takes the YAML as a **positional**, not ``--config``. Exit 0 pass / 1 warnings / 2 errors.
   * - ``spectramr campaign``
     - —
     - —
     - Manage campaigns (``submit`` / ``status`` / ``evaluate`` / ``cancel``).
   * - ``spectramr hpo``
     - yes
     - yes
     - Optuna-backed HPO over a base training YAML; each trial is a separate trainer subprocess.
   * - ``spectramr report``
     - yes
     - —
     - Canonical figures + tables for an output dir (same pipeline as the end-of-training hook).
   * - ``spectramr meta-evaluate``
     - —
     - —
     - Rank a metric set with the meta-evaluation framework.
   * - ``spectramr audit-ksd``
     - —
     - —
     - Tier-3 KSD defensibility audit over a generative-prior model.
   * - ``spectramr infer-protocol``
     - —
     - —
     - Posterior-mode inference over acquisition parameters.
   * - ``spectramr simulate-acquisition``
     - —
     - —
     - Synthetic IB-acquisition trajectory with per-step metrics; writes CSV.
   * - ``spectramr design-mrf-sequence``
     - —
     - —
     - Beltrami-CRLB-optimal MRF pulse-sequence design.
   * - ``spectramr regulatory``
     - —
     - —
     - Regulatory bundle CLI (``bundle`` / ``verify`` / ``status``).
   * - ``spectramr launch``
     - —
     - —
     - Unified launcher — any pipeline, anywhere. See `Launch backends`_.

Launch backends
---------------

``spectramr launch`` runs any pipeline anywhere: ``--where {local,docker,apptainer,slurm}``.
:doc:`execution_modes` describes the axes and the resource flags — verified against
the current code while writing this page — so what follows is only the part that
belongs here: **how the environment reaches the run**, which differs per backend and
is not documented elsewhere.

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
     - Docker inherits nothing. Image from ``SPECTRAMR_DOCKER_IMAGE``
       (default ``spectramr:v6.1``). See the warning below.

.. warning::

   **A docker run does not see your** ``SPECTRAMR_*`` **configuration.** Only the
   ``SPECTRAMR_LAUNCH_*`` provenance variables are forwarded, so ``SPECTRAMR_DATA_ROOT``
   is unset inside the container and ``env.data_root()`` falls back to
   ``./databases`` — a *different data root* than the identical command run locally,
   with no warning. ``scripts/container/entrypoint.sh`` exists to fix this by sourcing
   a mounted :file:`.env`, and :doc:`CLUSTER_DATA_LAYOUT` calls it "the Docker /
   Apptainer entrypoint", but :file:`scripts/container/Dockerfile` sets
   ``ENTRYPOINT ["python3.12", "-m", "spectramr.cli"]`` and never invokes it. Tracked
   in issue #1117. Until it is resolved, pass what you need explicitly.

``--dry-run`` prints the exact command or script without executing it, and is the
way to check what a backend actually forwards:

.. code-block:: console

   $ export SPECTRAMR_DATA_ROOT=/data/mine
   $ spectramr launch X.yaml --where docker --gpus 2 --dry-run
   docker run --gpus 2 --env SPECTRAMR_LAUNCH_BACKEND=docker ... \
     -v /repo:/workspace spectramr:v6.1 train --config X.yaml
   # note: SPECTRAMR_DATA_ROOT is absent

Distributed runs
----------------

``spectramr train-distributed`` is **not** launched directly — it expects a ``torchrun``
rendezvous. Two supported routes:

Single node
   .. code-block:: bash

      torchrun --nproc_per_node=N -m spectramr.cli train-distributed --config X.yaml

Multi-node (SLURM)
   :file:`scripts/training/train_distributed.sbatch` stands up the allocation and the
   rendezvous, one ``torchrun`` per node (``ntasks-per-node=1``; ``torchrun`` forks the
   per-GPU workers):

   .. code-block:: bash

      srun --kill-on-bad-exit=1 torchrun \
          --nnodes="${SLURM_NNODES}" \
          --nproc_per_node="${GPUS_PER_NODE}" \
          --rdzv_backend=c10d \
          --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
          --rdzv_id="${SLURM_JOB_ID}" \
          -m spectramr.cli train-distributed --config "${CONFIG}"

``nproc_per_node × nnodes`` is the DDP world size. ``RANK`` / ``LOCAL_RANK`` /
``WORLD_SIZE`` / ``MASTER_ADDR`` / ``MASTER_PORT`` are then set by ``torchrun`` and
read by the distributed pipeline — see `Distributed training`_ below.

.. note::

   ``launch --where slurm`` emits plain ``python -m spectramr.cli train``, **not**
   ``torchrun`` — the unified launcher does not wrap distributed runs. For DDP use
   the sbatch wrapper or ``torchrun`` directly.

:doc:`distributed_training` covers the strategy/parallelism settings themselves.

Scope: what counts as an environment variable here
--------------------------------------------------

**211 variables** are documented below, measured from the tracked tree rather
than transcribed from the two partial lists that already exist
(:file:`src/spectramr/core/env.py` and :file:`.env.example`) — see `Reconciliation`_
for where those two disagree with the code.

A shell script is full of names that *look* like environment variables but are
script-local plumbing (``BLUE``, ``CONFIG_FILE``, ``AUDIT_EXIT_CODE``). **247 such
names were excluded.** A name is documented here only when it is one of:

* read in Python via ``os.environ`` / ``os.getenv`` (including through a local
  ``FOO_ENV = "FOO"`` constant, which a naive grep for the literal misses);
* expanded in shell with the ``${VAR:-default}`` / ``${VAR:?}`` idiom, which is the
  unambiguous "may come from the environment" form;
* a scheduler-, launcher- or tool-provided name (``SLURM_*``, ``CUDA_*``, ``PYTEST_*``);
* an overridable ``?=`` assignment in a :file:`Makefile`.

.. note::

   These counts are a **reading, not a constant** — the same caveat CLAUDE.md
   applies to the model and strategy counts. Re-measure rather than quoting this
   page; the census script is described under `Re-measuring`_.

How a value reaches the framework
---------------------------------

Three surfaces are involved, and they are **not** equivalent:

:file:`.env.example`
   A copy-to-:file:`.env` template. :file:`.env` is gitignored and is sourced by the
   Makefile targets, the container entrypoint, some CI scripts and the multi-node
   wrapper :file:`scripts/training/train_distributed.sbatch`. It advertises
   **37** of the 211 variables.

   Sourcing is a **shell** contract, not a Python one — nothing in the package calls
   ``load_dotenv()``. A launcher that does not source :file:`.env` never sees it, so
   any hand-rolled ``srun ... torchrun`` needs
   ``set -a; source "${REPO_ROOT}/.env"; set +a`` of its own. The coverage is
   **not** uniform and the enumeration above rots — ask the tree:

   .. code-block:: bash

      grep -rl 'source.*\.env' Makefile scripts/

   The six one-off :file:`scripts/training/submit_*` scripts are absent from that
   list; a :file:`.env` has no effect on anything they launch.

:file:`src/spectramr/core/env.py`
   Name constants plus parsed-default helpers (``env.data_root()``,
   ``env.force_cpu()``, ``env.as_bool()``). Its docstring calls itself the single
   source of truth for *every* variable the framework reads; it currently holds
   **31**.

:file:`src/spectramr/infrastructure/config/env_resolver.py`
   Resolves the cache root and device preference at startup, and *writes*
   ``SPECTRAMR_CACHE_ROOT`` back into the environment so child processes inherit it.

.. warning::

   Reading a variable directly with ``os.environ.get("SPECTRAMR_...")`` instead of
   through :mod:`spectramr.core.env` is what produced the drift in `Reconciliation`_.
   New knobs belong in ``core/env.py`` **and** :file:`.env.example`, per
   non-negotiable #8 (every exposed knob is read, validated and stamped).

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
     - ``str(PROJECT_ROOT``
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
   * - ``SPECTRAMR_FIGURE_STYLE``
     - ``editorial``
     - Figure style for sim2rank plots.
       Read by ``scripts/sim2rank/style.py``
   * - ``FORCE_COLOR``
     - *unset*
     - Force ANSI colour in the logging service.
       Read by ``src/spectramr/infrastructure/services/logging_service.py`` (+2 more)
   * - ``NO_COLOR``
     - *unset*
     - Disable ANSI colour (the ``no-color.org`` convention).
       Read by ``src/spectramr/infrastructure/services/logging_service.py`` (+2 more)

Execution backends (``spectramr launch``)
---------------------------------------

The ``SPECTRAMR_LAUNCH_*`` trio is *written* by ``launch`` onto the child run so its provenance records the backend and resources it ran under.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``SPECTRAMR_DOCKER_IMAGE``
     - ``_DEFAULT_DOCKER_IMAGE``
     - Docker image for the ``docker`` execution backend.
       Read by ``src/spectramr/infrastructure/execution/backends.py``
   * - ``SPECTRAMR_APPTAINER_SIF``
     - ``_DEFAULT_APPTAINER_SIF``
     - Apptainer ``.sif`` image for the ``apptainer`` backend.
       Read by ``src/spectramr/infrastructure/execution/backends.py``
   * - ``SPECTRAMR_SLURM_ACCOUNT``
     - ``_DEFAULT_SLURM_ACCOUNT``
     - SLURM account fallback for ``ResourceSpec`` so a non-default user does not fail at ``sbatch`` time.
       Read by ``src/spectramr/infrastructure/execution/backends.py``
   * - ``SPECTRAMR_SLURM_PARTITION``
     - *unset*
     - SLURM partition fallback for ``ResourceSpec``.
       Read by ``src/spectramr/infrastructure/execution/backends.py``
   * - ``SPECTRAMR_LAUNCH_BACKEND``
     - *unset*
     - Set \*by\* ``spectramr launch`` on the child run so provenance records the backend it ran under.
       Read by ``tests/unit/cli/test_launch.py``
   * - ``SPECTRAMR_LAUNCH_ACCOUNT``
     - *unset*
     - Set \*by\* ``spectramr launch`` — resolved account, stamped into child provenance.
       Read by ``tests/unit/cli/test_launch.py``
   * - ``SPECTRAMR_LAUNCH_GPUS``
     - *unset*
     - Set \*by\* ``spectramr launch`` — resolved GPU count, stamped into child provenance.
       Read by ``tests/unit/cli/test_launch.py``

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
     - ``4``
     - OpenMP threads. ``main.py`` hard-sets ``1``; the runners default to ``4``.
       Read by ``runners/run_kspace_cold_diffusion.py`` (+3 more)
   * - ``MKL_NUM_THREADS``
     - ``4``
     - MKL threads. ``main.py`` hard-sets ``1``.
       Read by ``runners/run_kspace_cold_diffusion.py`` (+3 more)
   * - ``OPENBLAS_NUM_THREADS``
     - ``4``
     - OpenBLAS threads. ``main.py`` hard-sets ``1``.
       Read by ``runners/run_kspace_cold_diffusion.py`` (+2 more)
   * - ``PYTORCH_CUDA_ALLOC_CONF``
     - *unset*
     - Allocator config; ``main.py`` writes ``expandable_segments:True,max_split_size_mb:512``.
       Read by ``src/spectramr/main.py`` (+4 more)
   * - ``CUDA_CACHE_MAXSIZE``
     - *unset*
     - CUDA JIT cache cap; written by ``main.py`` (2 GB).
       Read by ``src/spectramr/main.py`` (+1 more)
   * - ``CUDA_CACHE_CONFIG``
     - ``str(cache_root / "cuda_cache``
     - CUDA JIT cache dir; written under the cache root.
       Read by ``src/spectramr/accelerator.py`` (+3 more)
   * - ``TORCH_HOME``
     - ``str(cache_root / "torch_cache``
     - Torch hub cache; written under the resolved cache root.
       Read by ``src/spectramr/accelerator.py`` (+3 more)
   * - ``TORCH_METRICS_CACHE``
     - *unset*
     - torchmetrics cache dir; written by ``main.py``.
       Read by ``src/spectramr/main.py``
   * - ``TRITON_CACHE_DIR``
     - ``str(cache_root / "triton_cache``
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
   * - ``NUMBA_CACHE_DIR``
     - *unset*
     - Numba cache dir; written by the legacy preprocessing script.
       Read by ``scripts/preprocessing/preprocessing_legacy.py``

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

Experiment tracking
-------------------

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``WANDB_MODE``
     - ``disabled``
     - Set to ``disabled`` by several scripts so an offline sweep never contacts W&B.
       Read by ``scripts/data/diagnose_queue_shapes.py`` (+2 more)
   * - ``WANDB_SILENT``
     - *unset*
     - Silences W&B logging in verification scripts.
       Read by ``scripts/data/verify_experiment_data_compliance.py`` (+1 more)
   * - ``WANDB_CONSOLE``
     - *unset*
     - W&B console capture mode, set by verification scripts.
       Read by ``scripts/data/verify_experiment_data_compliance.py`` (+1 more)

Test and CI switches
--------------------

Read only under :file:`tests/` or by CI scripts; none affect a normal training run.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``SPECTRAMR_ALLOW_NO_TORCH``
     - *unset*
     - Lets the test conftest proceed when torch is unavailable.
       Read by ``tests/conftest.py``
   * - ``SPECTRAMR_UPDATE_ARCH_BASELINE``
     - *unset*
     - Rewrites the architecture-fitness baseline instead of asserting against it.
       Read by ``tests/architecture/_fitness_lib.py``
   * - ``SPECTRAMR_UPDATE_LOSS_KEY_GOLDEN``
     - *unset*
     - Rewrites the loss-key golden file instead of asserting against it.
       Read by ``tests/experiments/test_loss_weight_no_op_migration.py``
   * - ``CONTRACT_STRICT``
     - ``0``
     - Makes the model-registry contract test strict.
       Read by ``tests/contracts/test_model_registry.py``
   * - ``RUN_ACTUAL_TRAINING_TESTS``
     - *unset*
     - Opts the dummy-config smoke tests into real training.
       Read by ``tests/smoke/test_dummy_configs.py``
   * - ``RUN_DRY_RUN_TESTS``
     - *unset*
     - Opts the dummy-config smoke tests into dry-run mode.
       Read by ``tests/smoke/test_dummy_configs.py``
   * - ``TEST_CONFIG_VALIDATION``
     - *unset*
     - Opts into config-validation smoke tests.
       Read by ``tests/smoke/test_dummy_configs.py``
   * - ``HYPOTHESIS_MAX_EXAMPLES``
     - ``100``
     - Hypothesis example budget for the fuzz tiers (default 100).
       Read by ``tests/fuzz/loss_composition_fuzz/test_loss_composition_finite_gradient.py`` (+1 more)
   * - ``PYTEST_XDIST_WORKER``
     - *unset*
     - Set by pytest-xdist; used for per-worker thread pinning.
       Read by ``tests/conftest.py`` (+1 more)
   * - ``PYTEST_XDIST_WORKER_COUNT``
     - *unset*
     - Set by pytest-xdist; used for per-worker thread pinning.
       Read by ``tests/conftest.py`` (+1 more)
   * - ``PYTEST_TIMEOUT``
     - ``600``
     - Per-test timeout for the sharded CPU test array.
       Read by ``scripts/ci/run_cpu_tests_array.sbatch``
   * - ``SPECTRAMR_TEST_EVENT_LOG``
     - *unset*
     - JSONL path the ``pytest_result_collector`` plugin writes per-test events to. Exported per shard by the CPU test array.
       Read by ``scripts/ci/pytest_plugins/pytest_result_collector.py``
   * - ``SPECTRAMR_TEST_SHARD``
     - *unset*
     - Shard index stamped into each event by the same plugin.
       Read by ``scripts/ci/pytest_plugins/pytest_result_collector.py``

Makefile overrides
------------------

Overridable with ``make <target> VAR=value``.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``CONFIG``
     - ``experiments/active/dummy_gan.yaml``
     - Makefile: config passed to ``make train`` / ``make predict``.
       Read by ``Makefile``
   * - ``PYTHON``
     - ``$(if $(wildcard .venv/bin/python),.venv/bin/python,python3)``
     - Makefile: interpreter, auto-detected as ``.venv/bin/python`` when present.
       Read by ``scripts/ci/cluster_verify.sh``
   * - ``FUZZ_RUNS``
     - ``-1``
     - Makefile: fuzz iteration count (``-1`` = default).
       Read by ``Makefile``
   * - ``SKILL_ROOTS``
     - ``skill-health:``
     - Makefile: extra roots for the skill-health target.
       Read by ``Makefile``

Third-party and shell environment
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``PYTHONPATH``
     - *unset*
     - Import path; several scripts prepend the repo root.
       Read by ``run_eda.sh`` (+21 more)
   * - ``VIRTUAL_ENV``
     - *unset*
     - Active virtualenv, checked by a rerun helper.
       Read by ``scripts/ci/rerun_smoke_failures_20260509.sh``
   * - ``LD_LIBRARY_PATH``
     - *unset*
     - Native library path, extended by some cluster scripts.
       Read by ``scripts/ci/run_cpu_tests_array.sbatch`` (+3 more)
   * - ``FREESURFER_HOME``
     - ``/project/<allocation>/opt/freesurfer``
     - FreeSurfer install root for the brain-pipeline segmentation step.
       Read by ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``

Secrets
-------

.. danger::

   These carry credentials. Set them in your shell or a secret store — never in a
   committed file, and never in :file:`.env.example`. None of them currently appear
   in :file:`.env.example`, which is correct.

.. list-table::
   :header-rows: 1
   :widths: 26 16 58

   * - Variable
     - Default
     - What it does / where it is read

   * - ``SPECTRAMR_ZIP_PASSWORD``
     - *unset*
     - Password for encrypted external dataset archives. \*\*Secret — never commit a value.\*\*
       Read by ``scripts/data/extract_external_datasets.py``
   * - ``LIGHTNING_API_KEY``
     - *unset*
     - Lightning.ai API key for the GPU smoke job. \*\*Secret.\*\*
       Read by ``scripts/ci/run_lightning_gpu_test.py``
   * - ``LIGHTNING_USER_ID``
     - *unset*
     - Lightning.ai user id. \*\*Secret.\*\*
       Read by ``scripts/ci/run_lightning_gpu_test.py``
   * - ``MRIXFIELDS_ZIP_PASSWORD``
     - *unset*
     - Password for the mrixfields archive. \*\*Secret — never commit a value.\*\*
       Read by ``scripts/data/extract_external_datasets.py``
   * - ``LIGHTNING_JOB_NAME``
     - ``f"gpu-test-{int(time.time(``
     - Lightning.ai job name; defaults to a timestamped string.
       Read by ``scripts/ci/run_lightning_gpu_test.py``

sim2rank
--------

The metric meta-evaluation subsystem is knob-driven: **37 variables**, almost
all read by :file:`scripts/sim2rank/run_full_pipeline.sbatch`, which validates each one
in a pre-flight block and dies on an unrecognized value rather than degrading
(non-negotiable #3). ``SIM2RANK_MODE=3d`` is deliberately *not implemented* and
raises — it was a silent no-op once, and the pre-flight exists because of it.

.. note::

   :file:`tests/unit/scripts/test_env_knob_advertisement.py` pins this family against
   :file:`.env.example` in both directions: an advertised knob nothing reads fails,
   and the two knobs an audit found read-but-unadvertised must stay advertised.

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Variable
     - Default
     - Read by

   * - ``SIM2RANK_ADR_EPSILON``
     - *unset*
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_AXIS_BANK``
     - ``registry``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_BETTING``
     - ``1``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_BETTING_ALPHA``
     - ``0.05``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_BOOTSTRAP``
     - ``30``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_BUILD_REPORT``
     - ``1``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_BUNDLE``
     - *unset*
     - ``tests/unit/test_sim2rank_bundle_integrity.py``
   * - ``SIM2RANK_CATEGORY_CURVE_SCALE``
     - ``zscore``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_DATA_DIR``
     - ``databases/m4raw/data/multicoil_...``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_DEVICE``
     - ``cuda``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_EXCLUDE_RADIOMICS``
     - ``1``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_FIGURE_MANIFEST``
     - *unset*
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_FIGURE_STYLE``
     - ``editorial``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_GEN3``
     - ``1``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_GEN4``
     - ``1``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_INCLUDE_SUMMARY``
     - ``1``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_INPUT_MODE``
     - ``synthetic``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_LAMBDA_PENALTY``
     - *unset*
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_LIKERT_JSON``
     - *unset*
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_MAX_CONTRASTS``
     - *unset*
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_MODE``
     - ``2d``
     - ``scripts/sim2rank/sim2rank.py``
   * - ``SIM2RANK_NOVEL_META``
     - ``1``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_NOVEL_SEVERITIES``
     - *unset*
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_N_CLUSTERS``
     - *unset*
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_ODM``
     - ``1``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_SEED``
     - ``42``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_SEVERITY_YARDSTICK``
     - ``none``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_SNAPSHOT_AXES``
     - *unset*
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_SNAPSHOT_LIGHT``
     - ``0``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_SNAPSHOT_MOSAIC``
     - ``1``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_SNAPSHOT_PHANTOM_SIZE``
     - ``256``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_STYLE_GALLERY``
     - ``0``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_SUBJECTS``
     - ``8``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_SUMMARY_BUCKETS``
     - ``4``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_TIMESTEPS``
     - ``20``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_TRANSFER_BADGE``
     - ``0``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SIM2RANK_UNIFIED_DEGRADATIONS``
     - ``1``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``

sim2rank matrix driver
----------------------

Read by :file:`scripts/sim2rank/submit_sim2rank_matrix.sbatch`.

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Variable
     - Default
     - Read by

   * - ``META_CONTRASTS``
     - ``T1 T2 FLAIR``
     - ``scripts/sim2rank/submit_sim2rank_matrix.sbatch``
   * - ``META_DEVICE``
     - ``cuda``
     - ``scripts/sim2rank/submit_sim2rank_matrix.sbatch``
   * - ``META_INPUT_DIR``
     - ``databases/m4raw/data/multicoil_...``
     - ``scripts/sim2rank/submit_sim2rank_matrix.sbatch``
   * - ``META_MAX_CONTRASTS``
     - ``3``
     - ``scripts/sim2rank/submit_sim2rank_matrix.sbatch``
   * - ``META_MAX_SUBJECTS``
     - ``16``
     - ``scripts/sim2rank/submit_sim2rank_matrix.sbatch``
   * - ``META_OUT_ROOT``
     - ``experiments/results/sim2rank_ma...``
     - ``scripts/sim2rank/submit_sim2rank_matrix.sbatch``
   * - ``META_SEED``
     - ``42``
     - ``scripts/sim2rank/submit_sim2rank_matrix.sbatch``
   * - ``META_SEVERITIES``
     - ``64``
     - ``scripts/sim2rank/submit_sim2rank_matrix.sbatch``

SLURM
-----

Provided by the scheduler; the repo only *reads* them. Do not set these by hand —
with the exception of ``SLURM_ACCOUNT`` / ``SLURM_PARTITION`` / ``SLURM_MEM`` /
``SLURM_TIME``, which :file:`scripts/training/submit_experiment_array.sh` accepts as
submission-time overrides.

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Variable
     - Default
     - Read by

   * - ``SLURMD_NODENAME``
     - ``local``
     - ``scripts/ci/run_cpu_tests_array.sbatch``, ``scripts/training/dispatch_experiments.sbatch`` (+2 more)
   * - ``SLURM_ACCOUNT``
     - ``<your-slurm-account>``
     - ``scripts/training/submit_experiment_array.sh``
   * - ``SLURM_ARRAY_JOB_ID``
     - ``${SLURM_JOB_ID:-local``
     - ``scripts/ci/run_cpu_tests_array.sbatch``, ``scripts/training/dispatch_experiments.sbatch``
   * - ``SLURM_ARRAY_TASK_ID``
     - ``0``
     - ``scripts/ci/run_cpu_tests_array.sbatch``, ``scripts/sim2rank/submit_sim2rank_matrix.sbatch`` (+3 more)
   * - ``SLURM_CPUS_PER_TASK``
     - ``16``
     - ``download_datasets.sh``, ``mata_eval.sbatch`` (+9 more)
   * - ``SLURM_GPUS``
     - *unset*
     - ``scripts/training/submit_exp11_ema_warmup_ablation.sbatch``, ``scripts/training/submit_exp11_fpk_ablation.sbatch`` (+3 more)
   * - ``SLURM_GPUS_ON_NODE``
     - ``${SLURM_GPUS:-1``
     - ``scripts/training/dispatch_experiments.sbatch``, ``scripts/training/submit_exp11_ema_warmup_ablation.sbatch`` (+2 more)
   * - ``SLURM_GPUS_PER_NODE``
     - *unset*
     - ``scripts/training/train_distributed.sbatch``
   * - ``SLURM_JOB_GPUS``
     - ``${SLURM_GPUS_ON_NODE:-${CUDA_VI...``
     - ``scripts/preprocessing/preprocessing.sh``, ``scripts/training/dispatch_experiments.sbatch``
   * - ``SLURM_JOB_ID``
     - *unset*
     - ``download_datasets.sh``, ``experiments/hpo/run_hpo.sh`` (+11 more)
   * - ``SLURM_JOB_NAME``
     - *unset*
     - ``download_datasets.sh``, ``scripts/ci/rerun_smoke_failures_20260509.sbatch`` (+5 more)
   * - ``SLURM_JOB_NODELIST``
     - *unset*
     - ``scripts/training/train_distributed.sbatch``
   * - ``SLURM_JOB_PARTITION``
     - *unset*
     - ``scripts/training/submit_experiment_11_batch.sh``
   * - ``SLURM_MEM``
     - *unset*
     - ``scripts/training/submit_experiment_array.sh``
   * - ``SLURM_MEM_PER_NODE``
     - *unset*
     - ``download_datasets.sh``, ``scripts/training/submit_experiment_11_batch.sh``
   * - ``SLURM_NNODES``
     - *unset*
     - ``download_datasets.sh``, ``scripts/training/train_distributed.sbatch``
   * - ``SLURM_NODELIST``
     - *unset*
     - ``scripts/ci/rerun_smoke_failures_20260509.sbatch``, ``scripts/ci/run_all_tests.sbatch`` (+4 more)
   * - ``SLURM_PARTITION``
     - *unset*
     - ``scripts/training/submit_experiment_array.sh``
   * - ``SLURM_SUBMIT_DIR``
     - *unset*
     - ``download_datasets.sh``, ``experiments/hpo/run_hpo.sh`` (+6 more)
   * - ``SLURM_TIME``
     - *unset*
     - ``scripts/training/submit_experiment_array.sh``

Script-scoped variables
-----------------------

Each of these is read by one or two operational scripts and has no effect on a
normal ``spectramr train`` run. They are the ``VAR=value ./<script>.sh`` overrides
that the maintainers' cluster workflow relies on.

.. warning::

   **Most of the scripts named in this table are not part of this distribution.**
   Of the 28 distinct scripts the 110 rows below cite, 27 are absent from the
   published tree -- they are SLURM drivers, dataset-mirroring tools and figure
   pipelines that run against the maintainers' cluster. Setting one of these
   variables in a checkout of this project will do nothing, because there is
   nothing here to read it.

   The rows are kept rather than deleted because they document how the published
   results were produced, and a reader reproducing a study needs to know which
   knob was turned. Read the table as a record of that study, not as a list of
   settings available to you. :doc:`known_limitations` states the same thing with
   the counts and the command that reproduces them.

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Variable
     - Default
     - Read by

   * - ``ADR_EPSILON``
     - ``0.01``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``, ``scripts/sim2rank/submit_sim2rank.sbatch``
   * - ``ALLOW_UNREADABLE``
     - ``0``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``ANATOMY``
     - *unset*
     - ``download_datasets.sh``
   * - ``AUDIT_ARMS``
     - *unset*
     - ``scripts/ci/cluster_verify.sh``
   * - ``AUDIT_FLAGS``
     - *unset*
     - ``scripts/ci/smoke_test_vf_configs.sh``
   * - ``AXIS_BANK``
     - ``registry``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``, ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``CACHE_DIR``
     - ``/tmp/cache/preprocessing``
     - ``scripts/preprocessing/preprocessing.sh``, ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``CONCURRENCY``
     - ``8``
     - ``scripts/training/submit_experiment_array.sh``
   * - ``CONTRASTS``
     - ``AXT1 AXT2 AXT2FLAIR``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``, ``scripts/sim2rank/submit_sim2rank_matrix.sbatch``
   * - ``COORDINATOR_HOST``
     - *unset*
     - ``scripts/container/launch_multi_container.sh``
   * - ``DATA_DIR``
     - ``/project/<allocation>/<user>/gan_mr...``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``, ``scripts/sim2rank/run_full_pipeline.sbatch`` (+1 more)
   * - ``DATA_TREE_PATH``
     - ``data_tree.json``
     - ``scripts/preprocessing/preprocessing.sh``
   * - ``DEVICE``
     - ``cuda``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``, ``scripts/sim2rank/run_full_pipeline.sbatch`` (+2 more)
   * - ``DISPATCH_DIR``
     - ``${REPO_ROOT``
     - ``scripts/training/dispatch_experiments.sbatch``, ``scripts/training/submit_experiment_array.sh``
   * - ``DRYRUN``
     - ``0``
     - ``scripts/training/dispatch_experiments.sbatch``, ``scripts/training/submit_exp11_ema_warmup_ablation.sbatch`` (+1 more)
   * - ``DRY_SUBMIT``
     - ``0``
     - ``scripts/training/submit_experiment_array.sh``
   * - ``EMIT_JOURNAL_FIGS``
     - ``0``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``EXPECT_INPLANE``
     - ``320``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``EXP_MANIFEST``
     - *unset*
     - ``scripts/training/dispatch_experiments.sbatch``
   * - ``FEDERATED``
     - ``0``
     - ``scripts/container/launch_multi_container.sh``
   * - ``FIGURE_MANIFEST``
     - *unset*
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``FILTER``
     - *unset*
     - ``scripts/ci/rerun_smoke_failures_20260509.sbatch``, ``scripts/ci/rerun_smoke_failures_20260509.sh``
   * - ``SPECTRAMR_ALLOW_MISSING_RIPGREP``
     - *unset*
     - ``tests/unit/config/test_schema_key_consumption.py``
   * - ``SPECTRAMR_FORENSICS_SCRIPT``
     - ``$HOME/.claude/skills/spectramr-im...``
     - ``scripts/ci/refresh_diagnostics.sh``
   * - ``SPECTRAMR_LAUNCH_``
     - *unset*
     - ``src/spectramr/infrastructure/execution/backends.py``
   * - ``SPECTRAMR_TEST_EVENT_MAXLEN``
     - *unset*
     - ``scripts/ci/pytest_plugins/pytest_result_collector.py``
   * - ``HEAD``
     - ``HEAD``
     - ``scripts/ci/cluster_verify.sh``
   * - ``IMAGE``
     - ``spectramr.sif``
     - ``scripts/container/launch_multi_container.sh``
   * - ``IM_SIZE``
     - ``256``
     - ``scripts/sim2rank/submit_sim2rank.sbatch``
   * - ``LAMBDA_PENALTY``
     - ``2.0``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``, ``scripts/sim2rank/submit_sim2rank.sbatch``
   * - ``LIKERT_JSON``
     - *unset*
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``MANIFEST``
     - ``${OUT_DIR``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``, ``scripts/training/submit_experiment_11_batch.sh`` (+3 more)
   * - ``MANIFEST_DIR``
     - ``/project/<allocation>/<user>/gan_mr...``
     - ``scripts/preprocessing/preprocessing.sh``
   * - ``MARKER_EXPR``
     - *unset*
     - ``scripts/ci/run_cpu_tests_array.sbatch``
   * - ``MAX_CONTRASTS``
     - ``3``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``, ``scripts/sim2rank/run_full_pipeline.sbatch`` (+2 more)
   * - ``MAX_SUBJECTS``
     - ``100``
     - ``scripts/ci/smoke_test_vf_configs.sh``, ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch`` (+2 more)
   * - ``MODE``
     - ``full``
     - ``scripts/ci/cluster_verify.sh``, ``scripts/ci/rerun_smoke_failures_20260509.sbatch`` (+3 more)
   * - ``NIFTI_DIR``
     - ``${OUT_DIR``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``NOVEL_SEVERITIES``
     - ``<default>``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``NO_AUDIT``
     - ``0``
     - ``scripts/training/dispatch_experiments.sbatch``, ``scripts/training/submit_experiment_array.sh``
   * - ``N_CLUSTERS``
     - ``6``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``, ``scripts/sim2rank/submit_sim2rank.sbatch``
   * - ``N_EXP``
     - *unset*
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``N_VOL``
     - *unset*
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``OUTPUT_DIR``
     - ``/project/<allocation>/<user>/gan_mr...``
     - ``run_eda.sh``, ``scripts/ci/run_ulf_hf_eda.sh`` (+5 more)
   * - ``OUT_DIR``
     - ``results/sim2rank_brain``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``PORT``
     - ``29500``
     - ``scripts/container/launch_multi_container.sh``
   * - ``PROD``
     - ``0``
     - ``scripts/training/dispatch_experiments.sbatch``, ``scripts/training/submit_experiment_array.sh``
   * - ``REFRESH``
     - ``0``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``REGION_MANIFEST``
     - *unset*
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``REPO_ROOT``
     - ``$SLURM_SUBMIT_DIR``
     - ``scripts/ci/check_layering.sh``, ``scripts/ci/refresh_diagnostics.sh`` (+15 more)
   * - ``RESUME``
     - ``0``
     - ``scripts/training/dispatch_experiments.sbatch``, ``scripts/training/submit_experiment_array.sh`` (+1 more)
   * - ``RESUME_FLAG``
     - ``false``
     - ``scripts/training/submit_experiments.sh``
   * - ``RUN_REGIONS``
     - ``0``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``SANITY_OVERRIDES``
     - *unset*
     - ``scripts/ci/smoke_test_vf_configs.sh``
   * - ``SEED``
     - ``4222``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``, ``scripts/sim2rank/run_full_pipeline.sbatch`` (+1 more)
   * - ``SEG_DIR``
     - ``${OUT_DIR``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``SELECTOR``
     - ``experiments/inprogress/vf/*.yaml``
     - ``scripts/training/submit_experiment_array.sh``
   * - ``SEVERITY_YARDSTICK``
     - ``none``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``, ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SNAPSHOT_AXES``
     - ``<all>``
     - ``scripts/sim2rank/run_full_pipeline.sbatch``
   * - ``SNAPSHOT_TIMESTEPS``
     - ``8``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``STEP``
     - ``all``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``TIMESTEPS``
     - ``20``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``, ``scripts/sim2rank/run_full_pipeline.sbatch`` (+1 more)
   * - ``TOP_K``
     - ``12``
     - ``scripts/sim2rank/run_fastmri_brain_pipeline.sbatch``
   * - ``TRAIN_ITERS``
     - ``5``
     - ``scripts/ci/rerun_failed_smoke.sh``, ``scripts/ci/rerun_smoke_failures_20260509.sbatch`` (+7 more)
   * - ``VENV_PATH``
     - ``${REPO_ROOT``
     - ``download_datasets.sh``, ``experiments/hpo/run_hpo.sh`` (+4 more)
   * - ``WORKER_HOSTS``
     - *unset*
     - ``scripts/container/launch_multi_container.sh``

Variables the framework writes
------------------------------

**32 variables are set by the framework itself**, which matters because a value
you export can be silently replaced. Two different disciplines are in use:

* ``os.environ.setdefault(...)`` — *yields* to a value you already set
  (:file:`accelerator.py`, :file:`infrastructure/distributed/launcher.py`,
  :file:`pipelines/train.py`).
* ``os.environ[...] = ...`` — **overwrites unconditionally**. :file:`main.py` does this
  at import time for the cache paths and for ``OMP_NUM_THREADS`` /
  ``MKL_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS``, which it pins to ``1``.

.. warning::

   Exporting ``OMP_NUM_THREADS=8`` before ``spectramr train`` has **no effect** —
   :file:`main.py` overwrites it with ``1`` on import. The ``4`` default you may have
   seen comes from :file:`runners/run_kspace_cold_diffusion.py`, a different entry
   point. Thread count is an entry-point property here, not an environment one.

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
     - ``str(cache_root / "cuda_cache``
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
     - ``4``
     - ``runners/run_kspace_cold_diffusion.py``, ``src/spectramr/main.py`` (+2 more)
   * - ``NUMBA_CACHE_DIR``
     - *unset*
     - ``scripts/preprocessing/preprocessing_legacy.py``
   * - ``OMP_NUM_THREADS``
     - ``4``
     - ``runners/run_kspace_cold_diffusion.py``, ``src/spectramr/main.py`` (+2 more)
   * - ``OPENBLAS_NUM_THREADS``
     - ``4``
     - ``runners/run_kspace_cold_diffusion.py``, ``src/spectramr/main.py`` (+1 more)
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
     - ``str(cache_root / "torch_cache``
     - ``src/spectramr/accelerator.py``, ``src/spectramr/main.py`` (+2 more)
   * - ``TORCH_METRICS_CACHE``
     - *unset*
     - ``src/spectramr/main.py``
   * - ``TRITON_CACHE_DIR``
     - ``str(cache_root / "triton_cache``
     - ``src/spectramr/infrastructure/config/env_resolver.py``, ``src/spectramr/infrastructure/services/memory_optimization_service.py``
   * - ``WANDB_CONSOLE``
     - *unset*
     - ``scripts/data/verify_experiment_data_compliance.py``, ``tests/integration/test_inprogress_yamls.py``
   * - ``WANDB_MODE``
     - ``disabled``
     - ``scripts/data/diagnose_queue_shapes.py``, ``scripts/data/verify_experiment_data_compliance.py`` (+1 more)
   * - ``WANDB_SILENT``
     - *unset*
     - ``scripts/data/verify_experiment_data_compliance.py``, ``tests/integration/test_inprogress_yamls.py``
   * - ``WORLD_SIZE``
     - ``-1``
     - ``src/spectramr/infrastructure/distributed/launcher.py``, ``src/spectramr/pipelines/distributed.py`` (+1 more)
   * - ``XDG_CACHE_HOME``
     - *unset*
     - ``src/spectramr/main.py``, ``src/spectramr/core/env.py``

Set by operational scripts
--------------------------

Shell scripts additionally ``export`` these unconditionally for their children.
They are *written*, not read from your environment, so exporting one yourself before
calling the script has no effect.

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Variable
     - Value
     - Exported by

   * - ``CUBLAS_WORKSPACE_CONFIG``
     - ``:4096:8``
     - :file:`scripts/ci/run_all_tests.sbatch`
   * - ``CUDA_VISIBLE_DEVICES``
     - *(varies)*
     - :file:`scripts/ci/run_cpu_tests_array.sbatch`
   * - ``SPECTRAMR_SUPPRESS_CLINICAL_WARNING``
     - ``1``
     - :file:`scripts/ci/run_cpu_tests_array.sbatch`
   * - ``MKL_NUM_THREADS``
     - ``8``
     - :file:`scripts/ci/run_all_tests.sbatch`
   * - ``NUMEXPR_NUM_THREADS``
     - ``$SLURM_CPUS_PER_TASK``
     - :file:`scripts/preprocessing/preprocessing.sh`
   * - ``OMP_NUM_THREADS``
     - ``8``
     - :file:`scripts/ci/run_all_tests.sbatch`
   * - ``PYTHONDONTWRITEBYTECODE``
     - ``1``
     - :file:`scripts/ci/rerun_smoke_failures_20260509.sh`
   * - ``PYTHONHASHSEED``
     - ``0``
     - :file:`scripts/ci/rerun_smoke_failures_20260509.sbatch`
   * - ``PYTORCH_CUDA_ALLOC_CONF``
     - ``max_split_size_mb:512``
     - :file:`scripts/preprocessing/preprocessing.sh`
   * - ``TORCHDYNAMO_DISABLE``
     - ``1``
     - :file:`scripts/ci/rerun_smoke_failures_20260509.sh`
   * - ``TORCH_COMPILE_DISABLE``
     - ``1``
     - :file:`scripts/ci/rerun_smoke_failures_20260509.sh`
   * - ``TORCH_USE_CUDA_DSA``
     - ``1``
     - :file:`scripts/preprocessing/preprocessing.sh`

Reconciliation
--------------

The two pre-existing lists disagree with the code, in one direction each.

**Variables are read in Python but absent from** :file:`core/env.py`, whose
docstring claims it lists every variable the framework reads. This gap was **44**
when the page was written. Twelve names — every ``SPECTRAMR_*`` variable read
inside :file:`src/spectramr/` — were registered subsequently and are now enforced
by ``test_every_spectramr_var_read_in_src_is_declared_here``, so that direction
cannot regress. Re-measure rather than quoting the total; the framework half that
*remains* is:

* ``SPECTRAMR_ALLOW_MISSING_RIPGREP``
* ``SPECTRAMR_ALLOW_NO_TORCH``
* ``SPECTRAMR_FIGURE_STYLE``
* ``SPECTRAMR_TEST_EVENT_LOG``
* ``SPECTRAMR_TEST_EVENT_MAXLEN``
* ``SPECTRAMR_TEST_SHARD``
* ``SPECTRAMR_UPDATE_ARCH_BASELINE``
* ``SPECTRAMR_UPDATE_LOSS_KEY_GOLDEN``
* ``SPECTRAMR_ZIP_PASSWORD``

These are read from :file:`scripts/` or :file:`tests/` rather than from the
package, which is why the guard does not reach them; they are test switches,
tracking flags and tooling knobs.

One family is in-package yet still unregistered, deliberately: ``SPECTRAMR_LAUNCH_*``
is composed at runtime from a prefix in
:file:`src/spectramr/infrastructure/execution/backends.py`, exported by
``spectramr launch`` and read back by the child run to stamp provenance. It is an
internal handoff rather than an operator knob — setting one by hand falsifies the
record — and its membership derives from ``_LAUNCH_RESOURCE_FIELDS``, so a
hand-copied list here would be its own drift surface. The guard excludes it
structurally (a literal ending in ``_`` is a prefix, not a variable), not by
name.

Nothing in either group breaks — each has a working direct reader — but
``env.names()`` reports only the subset in ``__all__``, so an unregistered knob
is invisible to anything that enumerates the environment rather than reaching
for a name it already knows.

Which is a smaller blast radius than it sounds, and worth stating precisely,
because there are **two** registries and they are not connected:

* ``env.names()`` has exactly one consumer in the tree —
  :file:`scripts/release/print_env.py`.
* ``spectramr doctor`` does **not** use it. It iterates ``_ENV_KNOBS``
  (:file:`src/spectramr/cli/diagnostics.py`), a hand-maintained tuple that is a
  separate partial list.

So registering a variable in :file:`core/env.py` does *not* make it appear in
``spectramr doctor``; the twelve added here do not. Reconciling ``_ENV_KNOBS``
against ``names()`` is a further piece of work, deliberately not done here.

.. note::

   ``__all__`` is what ``names()`` is derived from, so it is simultaneously the
   export list and the registry. A constant declared in :file:`core/env.py` but
   left out of ``__all__`` is invisible to ``names()`` *and* to the
   ``.env.example`` advertisement test, which iterates it —
   ``SPECTRAMR_GPU_MEMORY_FRACTION`` sat in that hole with a validating resolver
   whose own docstring called it "registered in core/env.py".
   ``test_all_constants_are_exported`` now closes it.

.. note::

   :file:`src/spectramr/cli/diagnostics.py` keeps a *third*, hand-maintained list of 12
   names to display. All 12 are genuinely read, but it is neither a subset nor a
   superset of the other two — a fourth surface to keep in sync.

In the other direction, **every variable advertised in** :file:`.env.example` **does
have a reader** — verified against the same census, and consistent with the
advertisement test passing. There are no advertised-but-inert knobs.

Known rough edges
-----------------

* **Fixed (#1250).** :file:`src/spectramr/core/metrics/evaluation_metrics.py` used to do
  ``os.environ.setdefault("TMPDIR", "/tmp/<user>")`` at import — a **hardcoded username**
  as the fallback temp dir, pointing on any other account at a directory the user may not
  own. It also set ``TORCH_CUDA_EAGER_CACHE_MANAGER`` behind a ``torch.cuda.is_available()``
  probe, which initialised CUDA merely to import a metrics module. Both writes are gone; the
  module now writes no environment variable at all, and
  ``tests/unit/core/metrics/test_evaluation_metrics.py::TestImportWritesNoEnvironment``
  pins that structurally (an AST walk over the module body, plus a clean-subprocess import).
  Worth noting *why* it mattered beyond the username: ``TMPDIR`` is **tier 2** of
  ``resolve_cache_root``, so importing a metrics module silently redirected the
  framework-wide cache root, and whether it won depended on import order. It was also inert
  on its own terms — the assignment sat *below* that module's ``import torch``, and PyTorch
  reads these when the CUDA allocator initialises. :file:`src/spectramr/main.py` owns this
  bootstrap correctly, above ``import torch``. The remaining ``TMPDIR`` writers derive it
  from the resolved cache root.
* **Fixed (#1250).** :file:`src/spectramr/infrastructure/services/memory_optimization_service.py`
  used to ``makedirs("/tmp/triton_cache")`` and point ``TRITON_CACHE_DIR`` at it. Two
  defects: the path carried no ``$USER`` term (``/tmp`` is sticky, so the directory belongs
  to whoever created it first — the exact EACCES documented under `The cache block has exactly two knobs`_
  above), and it was a **second writer** to a variable ``configure_cache_environment()``
  already owns, so the effective value depended on whether a run had reached that service
  yet. The service now sets no cache directory; steer it with ``SPECTRAMR_CACHE_ROOT``.
* Three project-root spellings coexist — ``PROJECT_ROOT``, ``SPECTRAMR_DATA_ROOT`` and
  ``SPECTRAMR_ROOT`` — resolved by different modules with different precedence.
  ``env.project_root()`` checks the first two; ``PathNormalizer`` checks the third.

Re-measuring
------------

This page is hand-maintained, so treat it the way CLAUDE.md treats the registry
counts: re-measure, do not quote. The census that produced it classifies a shell
name as an environment *input* rather than script-local plumbing by this rule, which
is the part that is easy to get wrong:

#. A ``${VAR:-default}`` or ``${VAR:?}`` expansion **anywhere** makes it an input,
   outranking any assignment — because the env-prefix form ``VAR=$X cmd`` re-exports
   to a child rather than shadowing the name.
#. Comment lines are stripped before assignment detection. A header comment
   advertising ``#   SIM2RANK_MODE=2d`` is documentation, not an assignment.
#. Otherwise, a name assigned a literal anywhere in the shell corpus is script-local.

Skipping rule 1 hides 4 advertised sim2rank knobs; skipping rule 2 hides 29 more.
