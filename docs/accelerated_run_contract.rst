========================
Accelerated-run contract
========================

.. module:: spectramr.core.compute_device

**Heavy pipelines run on an accelerator. If none is available, they raise —
they never silently degrade to CPU.**

"Heavy" means anything that consumes real GPU-hours: training, sanity-check,
experiment, inference, prediction, validation, evaluation, HPO, ablation,
benchmarking, distributed runs, campaigns, PGGS reconstruction (Gaussian
splatting with latent-diffusion refinement and test-time optimisation) and the
Tier-2 audit probe. The set is :data:`HEAVY_PIPELINES`.

Why
===

A silent CPU fallback is invisible by construction, which is what makes it the
most expensive kind:

* A batch job that lands on a GPU-less node, or whose CUDA driver faults, would
  run to completion on CPU at roughly 100x slowdown — and **report success**.
  The wall-clock allocation is spent; a warning scrolls past in a 40-hour log.
* A device resolver that wraps its body in a bare ``except Exception`` and logs
  *"Falling back to CPU mode..."* turns a driver mismatch, an ECC fault and an
  OOM on the probe allocation alike into a *successful* CPU run.
* Flattening ``"auto"`` to ``"cpu"`` **before**
  :class:`~spectramr.infrastructure.services.device_manager.DeviceManager` is
  constructed means that class's own (correct) CUDA guard never fires on the
  live path.

This page is the standing rule at the hardware layer -- *a silent fallback is
forbidden: an unavailable or unknown option raises rather than degrading to a
default* -- together with its companion, *a warning is not an acceptable resting
state*.

The policy
==========

:func:`resolve_compute_device` is a **pure function** of the request, the
pipeline, and hardware availability, so every branch is unit-testable without a
GPU. :func:`resolve_torch_device` is the thin shell that queries torch and
delegates to it.

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - Requested
     - Accelerator present?
     - Outcome
   * - ``auto`` (or unset)
     - yes
     - CUDA (or MPS) — ``accelerated=True``
   * - ``auto`` (or unset)
     - no
     - **raise** :exc:`AcceleratorRequiredError` for a heavy pipeline
   * - ``cuda`` / ``cuda:N``
     - no
     - **raise** — always, even under ``FORCE_CPU``
   * - ``cpu``
     - n/a
     - CPU, ``cpu_opt_in=True`` — the user dictated it

An explicit ``cuda`` request on a host without CUDA is a *broken environment*,
not a preference, so it raises unconditionally. ``FORCE_CPU`` only relaxes the
``auto``-with-no-GPU case.

Opting into CPU (the user-dictated escape)
==========================================

CPU remains reachable, but only deliberately — never by accident:

* ``--device cpu`` on the CLI,
* ``run.device: cpu`` in the experiment YAML (the top-level ``device:``
  spelling was retired with a **raise** posture -- see ``RENAMES``),
* ``SPECTRAMR_DEVICE=cpu``,
* ``FORCE_CPU=true`` in the environment (the CI / co-simulation hatch).

Each path is logged loudly and stamped into the resolved
:class:`DeviceDecision` (``source``, ``cpu_opt_in``), so provenance records
*why* a run was not accelerated.

.. note::

   ``provenance.json`` records the **resolved** device, not the requested one.
   ``collect_run_provenance`` runs before the training environment exists, so
   the only device it can see is the CLI's ``--device`` — ``None`` whenever the
   caller omits it, which the SLURM dispatcher always does.
   ``pipelines/train.py`` therefore back-fills ``provenance["device"]`` from
   ``pipeline.device`` once the environment is built. Without that back-fill the
   record stamps ``"device": null`` while training on a GPU, and cannot answer
   the one question it exists for.

.. warning::

   ``FORCE_CPU`` is validated against the accepted boolean spellings and
   **raises** on anything else (``FORCE_CPU=maybe`` fails at startup). An
   advertised knob that silently ignores an unknown value is a defect.

Which device the launch surface asks for
========================================

:func:`resolve_compute_device` can only honour what it is handed, so the
*launch* surface has to hand it the user's actual request. ``--device`` defaults
to ``None`` on every verb but the two pinned exceptions below, ``infer``,
``infer-dataset``, ``predict`` and ``hpo`` included: a non-``None`` argparse
default makes the config's own device unreachable by construction — a knob the
schema advertises with no consumer on those paths.

The resolution order for a launch verb is:

#. the CLI's ``--device`` (absent unless the caller passes it),
#. ``training.device`` in the YAML,
#. ``run.device`` -- **only when the YAML declared it**,
#. otherwise ``None``, which :func:`resolve_compute_device` reads as ``auto``.

Step 3 is not a plain attribute read. ``run.device`` carries a schema default
of ``"cuda"``, and the resolver treats an explicit ``"cuda"`` as a hard
requirement that ``FORCE_CPU`` may not relax while ``auto`` is relaxable --
so substituting the default for a declaration would quietly convert every
undeclared arm into a CUDA-mandatory one. :func:`spectramr.main._declared_device`
is the single owner of that distinction; it gates on
Pydantic's ``model_fields_set`` and returns ``None`` when the key is absent.

**One resolution, every consumer.** A verb resolves the device once and must
then hand *that* value to both halves of the run: the accelerator it arms and
the pipeline (or container) it launches. Passing the resolved device to one and
the raw ``--device`` argument to the other reintroduces the divergence this
contract exists to remove, and it does so invisibly -- both values are
well-formed device strings, so nothing raises. Treat the resolved value as the
only device string in scope after resolution; a second read of the argument or
the config below that point is the defect.

Two verbs keep a pinned device on purpose and are exempt: ``meta-evaluate``
defaults to ``auto``, and ``design-mrf-sequence`` to ``cpu``. Sim2Rank's
documented CPU backing is the same deliberate exception.

Audit: the Tier-2 acceleration gate
===================================

``spectramr audit <yaml> --probe`` **checks that the run is accelerated by a
device other than CPU**, and fails otherwise.

This closes a facade. The Tier-2 probe's headline value is catching CUDA OOM at
the configured batch/patch size and AMP / GradScaler double-unscale traps — and
**neither exists on CPU**. A probe that runs on CPU certifies "did not crash on
CPU", never "this arm will run on the GPU it was scheduled for".

The gate (``_gate_probe_acceleration`` in :mod:`spectramr.cli.app`) resolves the
probe device as **CLI ``--device`` > the config's own ``training.device`` > a
*declared* ``run.device`` > ``auto``** -- so by default the probe exercises the
device the *real run* will use -- and then emits a ``tier2_probe_accelerated``
check. It shares :func:`spectramr.main._declared_device` with the launch verbs:
reading ``run.device`` as a plain attribute resolved to ``"cuda"`` for every
arm that declared no device, which made the ``auto`` leg unreachable and the
"CPU, user dictated" row below unreachable under ``FORCE_CPU``.

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Probe device
     - Severity
     - Meaning
   * - CUDA / MPS
     - ``info`` (pass)
     - Real probe: OOM + AMP coverage active
   * - CPU, user dictated
     - ``info`` (pass)
     - Degraded probe; OOM/AMP coverage **disabled**, recorded as such
   * - CPU, no accelerator
     - ``error`` (**fail**)
     - Audit exits 2; the probe is skipped rather than run as a facade

The JSON report stamps ``tier_2.accelerated`` and ``tier_2.device_source`` so
triage tooling can distinguish a real probe from a degraded one after the fact.

A CPU probe by explicit request is deliberately **not** a warning: warnings must
be actionable, and "you asked for CPU" is not. It is recorded, not nagged about.

Expect new Tier-2 failures — they are real
------------------------------------------

.. important::

   The first smoke run after this change will likely surface **new Tier-2
   failures**, and they are almost certainly true positives.

   Until now the probe ran on CPU, so it never allocated at the configured
   batch/patch size on the GPU and never exercised AMP / GradScaler on CUDA. It
   now does both. Any arm whose batch size genuinely OOMs, or whose AMP path
   genuinely double-unscales, has been passing Tier-2 on a technicality.

   A jump in Tier-2 failures is the gate **working**, not a regression. Read it
   as "these arms were always broken and the probe was blind", not "the new
   check is too strict". Judge each failure on its merits before relaxing
   anything — and never relax it with a blanket ``--device cpu``, which would
   restore exactly the facade this closes.

   (The internal wrapper that sweeps arms in bulk already runs on a GPU host, so
   the gate does not break it. That wrapper is not part of this release; what
   survives here is the rule it follows -- a bulk runner passes no ``--device``
   and lets the probe follow each arm's own configured device.)

Call sites brought under the contract
=====================================

Every resolver that decides **whether a run is accelerated** now delegates to the
SSOT:

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Site
     - Silent fallback removed
   * - ``accelerator.initialize_accelerator``
     - ``"CUDA not available. Falling back to CPU."``
   * - ``bootstrap.build_container``
     - unset → ``cpu``; ``auto`` → ``cpu`` (this one made
       ``DeviceManager``'s correct CUDA guard unreachable)
   * - ``cli.app._resolve_device``
     - ``auto`` → ``cpu``
   * - ``cli.app.benchmark``
     - ``"cuda" if is_available() else "cpu"`` (CPU timings tabulated against
       GPU baselines)
   * - ``services.device_policy.DevicePolicy``
     - ``fallback_to_cpu=True`` **by default**
   * - ``training.builders.environment._resolve_device``
     - ``auto`` → ``cpu``; ``None`` → ``cpu``
   * - ``training.builders.director._resolve_device``
     - CUDA-unavailable → ``cpu``
   * - ``training.utils.initialize_device``
     - bare ``except Exception`` → ``cpu`` (**the worst**: any CUDA fault became
       a "successful" CPU run)
   * - ``pipelines.infer.run_inference_pipeline``
     - bare ``torch.device(device)`` — no ``auto``, no availability check
   * - ``orchestration.campaign_evaluator``
     - ``"cuda" if is_available() else "cpu"`` on the per-sample metric pass
   * - ``config.env_resolver.resolve_device``
     - ``SPECTRAMR_DEVICE`` defaulted to ``"cpu"``

Known residue
-------------

Roughly 30 ``"cuda" if torch.cuda.is_available() else "cpu"`` one-liners remain
under ``models/``, ``infrastructure/services/metrics_tracker``,
``performance_optimizer``, and ``infrastructure/distributed/``. These are
**downstream per-component defaults**, not entry-point decisions: they run only
after an entry point has already resolved (and, on a heavy run, *validated*) the
device, so they can select CPU only when the user opted in. ``DeviceManager``'s
``preferred_device=None`` branch is likewise reachable only from direct
construction in test fixtures — the live DI path always passes a concrete,
contract-resolved device. They are latent, not load-bearing; tightening them is
a follow-up, not a prerequisite.

Device capabilities
===================

Resolving *which* device a run lands on is only half the question; the other
half is what that device can do. ``spectramr.core.device_capabilities`` answers
it, and ``compute_device`` stays the owner of the first half.

The capability that matters today is **native bfloat16**, and it carries a trap
worth stating plainly. ``torch.cuda.is_bf16_supported()`` takes
``including_emulation=True`` by default, and that branch only checks a bf16
tensor can be *created*. Measured on an sm_75 card::

    torch.cuda.is_bf16_supported()                           # -> True
    torch.cuda.is_bf16_supported(including_emulation=False)  # -> False

The target clusters are V100s (sm_70) — which is why torch is pinned to the
cu126 lane at all, since it is the last one shipping ``sm_70`` kernels. A gate
written against the default probe would therefore *confirm* bf16 on exactly the
hardware it needed to reject, and leave a paper trail saying the configuration
was checked. The test is the compute capability instead: native bf16 needs
``sm_80`` or later.

Below that, torch still runs bf16 by emulating it — slower than fp16 and
numerically unlike the declaration. So it raises rather than downgrading;
an automatic downgrade to fp16 would be worse still, because
``get_autocast_context`` turns fp16 into a ``nullcontext`` for complex arms and
would quietly produce fp32 on the arms this framework cares most about.

The gate has two halves, and the split is deliberate:

``mixed_precision.assert_amp_dtype_supported``
    The enforcer. Runs where the device is real, and raises.

``ConfigHealthChecker.check_bf16_requires_ampere``
    The early warning. ``audit`` is ``--strict`` and warnings exit 2, while the
    audit legitimately runs on a login node whose GPU differs from the compute
    node's — so an *unverifiable* answer reports at ``info`` and never gates.
    Only a capability this machine could actually read produces an error. Set
    ``SPECTRAMR_TARGET_COMPUTE_CAPABILITY`` to make it decidable there.

Every resolved capability is stamped into run provenance under
``device_capabilities``, alongside the ``source`` it came from — distinct from
``torch_runtime``'s hardware *inventory*, which lists what was visible rather
than what the run concluded.

API
===

.. autoexception:: AcceleratorRequiredError
.. autoclass:: DeviceDecision
   :members:
.. autofunction:: resolve_compute_device
.. autofunction:: resolve_torch_device
.. autofunction:: cpu_opt_in_from_env
.. autodata:: HEAVY_PIPELINES

See also
========

* :doc:`known_limitations` — what the shipped tree does not do, measured rather
  than estimated.
* ``spectramr audit --help`` — the Tier 0/1/2 ladder this gate extends. The
  ladder's own user guide belongs to the internal documentation tree and is not
  published with this release.
