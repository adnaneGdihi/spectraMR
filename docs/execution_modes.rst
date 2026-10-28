Execution Modes
===============

spectraMR separates *what* you run from *where* and *how many* — so the launch
surface is a small set of orthogonal axes rather than a pile of separate
"modes". Pick a value on each axis:

============== ====================================== ==============================================
Axis           Question                               Values
============== ====================================== ==============================================
WHAT           which config                           a training YAML (or a campaign manifest)
WHICH-PIPELINE which verb (must accept ``--config``)   ``train`` · ``sanity_check`` · ``infer`` · ``hpo`` · ``ablation`` · ``experiment``
WHERE          where it executes                      ``local`` · ``docker`` · ``apptainer`` · ``slurm``
HOW-MANY       one run or a sweep                     ``single`` · ``campaign``
============== ====================================== ==============================================

Plus a fourth, *imperative* surface that removes the WHAT axis entirely — you
write Python instead of a config: see :doc:`scripting_api`.

The unified launcher: ``spectramr launch``
------------------------------------------

``launch`` addresses the cube directly. The dedicated commands
(``spectramr train``, ``spectramr campaign submit``, ``sbatch …``) still work —
``launch`` is an additive front door over the same machinery.

.. code-block:: bash

   # WHAT × WHICH-PIPELINE × WHERE × HOW-MANY
   spectramr launch X.yaml                                   # train, local, single (defaults)
   spectramr launch X.yaml --where slurm --gpus 2            # train on SLURM (2 GPUs)
   spectramr launch X.yaml --where docker                    # train in a Docker container
   spectramr launch X.yaml --pipeline infer --where local \
       -- --checkpoint best.safetensors --input d/ --output o/      # a different verb; '--' passes verb args
   spectramr launch C.yaml --fanout campaign --where slurm   # a whole campaign

``--dry-run`` prints the exact command / sbatch script that *would* run without
executing it — useful for inspecting the resolved resources:

.. code-block:: bash

   $ spectramr launch X.yaml --where slurm --gpus 2 --mem 64G --dry-run
   #!/bin/bash
   #SBATCH --job-name=spectramr-train
   #SBATCH --account=<your-slurm-account>
   ...
   #SBATCH --gpus=2
   python -m spectramr.cli train --config X.yaml

Resource flags — ``--account`` · ``--partition`` · ``--mem`` · ``--gpus`` ·
``--time`` · ``--nodes`` — populate a single ``ResourceSpec``. ``account`` and
``partition`` fall back to ``$SPECTRAMR_SLURM_ACCOUNT`` / ``$SPECTRAMR_SLURM_PARTITION``,
so a non-default user need not edit anything to submit.

Job arrays
~~~~~~~~~~

``ResourceSpec.array`` (a SLURM array spec like ``"0-9"``, ``"0-19%4"``, or
``"1,3,5"``) adds a ``#SBATCH --array=…`` line directly after ``--gpus`` — the
single source of truth for fanning one script out over a manifest of arms. It is
``None`` by default, so every non-array caller renders a byte-identical
directive block; an empty/whitespace spec is rejected rather than emitting an
invalid ``--array=``.

The per-task body — resolve the config for THIS array index from a manifest,
the ``TRAIN_ITERS`` smoke cap, the Tier-0/1 audit pre-flight, then ``train`` —
is the torch-free SSOT :mod:`spectramr.cli.manifest_dispatch`. A SLURM array
task body is therefore one line::

   python -m spectramr.cli.manifest_dispatch --manifest M.txt --index $SLURM_ARRAY_TASK_ID \
       [--train-iters N] [--resume] [--no-audit] [--dry-run]

The ``TRAIN_ITERS`` cap is a ``yaml.safe_load`` parse there rather than a shell
text pipeline: ``safe_load`` strips inline comments natively and a missing
``eval_interval`` is just ``None``, so there is no ``set -e`` foot-gun that can
abort an array task mid-fan-out. The shell keeps only the REPO_ROOT spool anchor
and the fail-loud venv/torch probe (their regression tests run torch-free and
must not reach Python before the venv); the ``--dry-run`` path is torch-free, so
it runs the unit tests without a GPU.

An arm whose registered model class declares ``parameter_free`` (a classical solver such
as ``cs_l1_wavelet_sense``) has nothing to train, and ``train`` refuses it. The task
body asks the class (:func:`~spectramr.cli.manifest_dispatch.declares_parameter_free_arm`)
and launches such an arm with ``--validate-only`` and no ``--resume``, so it stays in the
array beside the learned arms and writes to the same run tree. A model type the registry
does not know raises; it is never read as "learned".

**Runs follow** ``dev``. A manifest lists config *paths*, and each task reads its
YAML and imports ``src/`` from the live checkout when it starts. Both submitters
(``submit_experiment_array.sh`` and the campaign array path) refuse a checkout
that is not on the ``dev`` branch, and write ``<manifest>.snapshot.json`` beside
the manifest, holding HEAD, the branch, its dirty flag and a sha256 per listed YAML
(:mod:`spectramr.core.dispatch_snapshot`, git helpers in
:mod:`spectramr.core.dispatch_git`). Every task start, a requeued wall-clock link
included, checks the live checkout and exits 2 naming each problem:

- the checkout is on another branch or a detached HEAD;
- dev no longer contains the submitted commit (rewound, or another line);
- the task's YAML differs from the record **and** from its committed version
  (an uncommitted edit).

Dev advancing past the submitted commit is not drift: a task that starts after a
``git pull`` of dev runs the pulled code and YAML, and the commit it actually ran
is stamped into the run record under ``"dispatch_snapshot"``
(``recorded_head`` against ``current_head``). One array can therefore span
several dev commits; read ``current_head`` per run when comparing arms.
``SPECTRAMR_DISPATCH_ALLOW_DRIFT=1``, set when the array is submitted (it rides
``--export=ALL``), lets a refused task run anyway, with the drift recorded. The
dirty flag is recorded but not compared, so uncommitted edits to tracked code are
not covered. A manifest written before the record existed has none; its tasks
print a warning and proceed.

Provenance
----------

When a run is started through ``spectramr launch``, the resolved backend +
``ResourceSpec`` are handed to the child via ``SPECTRAMR_LAUNCH_*`` environment
variables (inherited in-process, by SLURM through ``--export``, and forwarded
into containers with ``--env``). The run's ``run_summary.json`` then records
them under a ``"launch"`` block — so a result is traceable to *where* and *with
what resources* it ran, not just *what* config. A plain ``spectramr train`` (not
launched) leaves the block ``null``.

Choosing a WHERE
----------------

- **local** — runs in *this* process (no subprocess), the fastest path for
  development and the default.
- **docker** — ``docker run --gpus <N> -v $PWD:/workspace <image> <verb> …``;
  image overridable via ``$SPECTRAMR_DOCKER_IMAGE``.
- **apptainer** — ``apptainer run --nv --env CUDA_VISIBLE_DEVICES=0,…,N-1
  --bind $PWD:/workspace <sif> <verb> …`` for HPC; sif overridable via
  ``$SPECTRAMR_APPTAINER_SIF``.
- **slurm** — generates and submits an ``#SBATCH`` script from the
  ``ResourceSpec``.

.. note::

   The ``--gpus`` *count* is honoured by **every** backend, default ``1``:
   **slurm** emits a generic ``#SBATCH --gpus=N`` (no GPU type pinned, so the
   chosen account/partition/cluster decides the hardware); **docker** emits
   ``--gpus N`` (the container is
   re-numbered to see exactly N devices); **apptainer** emits ``--nv`` plus
   ``CUDA_VISIBLE_DEVICES=0,…,N-1``. The sentinel ``--gpus 0`` means **all
   visible GPUs** (``--gpus all`` / unrestricted ``--nv``). A negative count is
   rejected at startup.

Choosing a HOW-MANY
-------------------

- **single** — one run of one config.
- **campaign** — a manifest of many arms with comparative evaluation.
  ``launch --fanout campaign --where slurm`` (default)
  submits each arm as an sbatch job; ``--where docker`` / ``apptainer`` run each
  arm in a container (synchronously, parallel-mode campaigns only). ``--where
  local`` is not supported for campaigns.

.. seealso::

   * :doc:`scripting_api` — the imperative ``fit`` / ``Trainer`` surface.
   * :doc:`plugins` — custom components defined outside the source tree.
   * :doc:`cli_reference` — the full per-command reference.
