=================
Known limitations
=================

This page lists behaviour that is real, reproducible, and *not* what a reader
would reasonably assume from the surrounding documentation. It exists because a
documented limitation is honest and a documented feature that is inert is not.

Everything here was measured on the shipped package, not inferred from a plan or
an audit note. Where a number appears, the command that produced it appears with
it, because these counts drift.

.. note::

   This page is **hand-maintained**, not generated. Nothing recomputes it, so an
   entry can outlive the limitation it describes. Where an entry gives a command,
   that command -- not this page -- is the current answer.

Documentation
=============

Commands in the documentation are checked against the tree, prose references are not
-------------------------------------------------------------------------------------

``scripts/ci/check_docs_paths_exist.py`` fails on any *pasteable* command that
names a repository path this tree does not contain — across ``docs/`` and the
root-level pages alike. This tree passes it with **zero command findings**, and
building the distribution runs the same check, so a release cannot ship one.

Prose references are reported by the same gate, and do not block a release: the
distribution build runs it as ``--commands-only``, which fails on commands alone.
Run without that flag -- the form printed below -- it counts prose references too
and exits nonzero, so a nonzero exit there is the report, not a broken tree.

They fall in two classes. One is provenance: a sentence naming where a piece of
code lives, accurate about the repository it lives in but not a path you can open
in this one. Those were left rather than stripped, because removing them loses
the provenance without giving a reader anything to act on. The other is a genuine
gap -- a page describing a maintainer workflow whose helper script is not part of
this distribution.

So read a prose path as a pointer rather than as something you can run, and check
it against the tree before relying on it. The gate prints the current list::

    python scripts/ci/check_docs_paths_exist.py

Configuration
=============

Unknown keys inside an ``extra="ignore"`` block are dropped in silence
----------------------------------------------------------------------

The configuration schema is mostly strict: of the 339 schema classes reachable
from a config, **252 are** ``extra="forbid"`` and reject an unknown
key outright. But **66 are** ``extra="ignore"``, and inside one of those a key
the schema does not declare is discarded with no error and no warning. The YAML
still shows it; the run never sees it; the arm trains on the default.

This is the failure mode to know about when a knob you set appears to do
nothing. It is not detectable by reading the config back, because the key is
gone by then.

.. code-block:: console

   $ python scripts/ci/report_discarded_config_keys.py <dir>

That script reports every such key across a directory of configs. Run it on your
own configs; a clean report is a meaningful result, and a non-empty one names the
block and the key. Pass the directory explicitly -- it defaults to
``experiments/inprogress``, which is an internal path that does not exist in this
distribution.

To re-measure the class counts above:

.. code-block:: python

   import typing
   from pydantic import BaseModel
   from spectramr.config.schemas import training as training_schemas
   from spectramr.config.settings import TrainingSettings

   def models_in(annotation):
       """Unwrap typing constructs to a fixed point.

       A fixed depth is not enough. A discriminated union is annotated
       ``Optional[Annotated[A | B | ..., FieldInfo(discriminator=...)]]``, so its
       members first appear three ``get_args`` levels down -- and those are the
       polymorphic blocks most likely to carry loosely-validated keys.
       """
       found, stack = set(), [annotation]
       while stack:
           a = stack.pop()
           if isinstance(a, type) and issubclass(a, BaseModel):
               found.add(a)
           stack.extend(typing.get_args(a))
       return found

   seen, counts = set(), {}
   def walk(m):
       if m in seen:
           return
       seen.add(m)
       policy = m.model_config.get("extra") or "ignore"
       counts[policy] = counts.get(policy, 0) + 1
       for f in m.model_fields.values():
           for sub in models_in(f.annotation):
               walk(sub)

   # TWO roots. Field-reachability from TrainingSettings finds everything held
   # AS a field; it cannot reach a top-level paradigm schema, which a YAML
   # selects by `training_mode`. Seeding only the first root reports 8 open
   # classes where 20 are reachable.
   walk(TrainingSettings)
   for name in sorted(set(training_schemas._MODE_DISPATCH.values())):
       walk(getattr(training_schemas, name))

   print(sum(counts.values()), counts)
   print(sorted(m.__name__ for m in seen if m.model_config.get("extra") == "allow"))

Twenty-one schema classes accept *any* key
------------------------------------------

``extra="allow"`` is the opposite hazard: the key is not dropped, it is kept —
as an untyped extra field that nothing validates and nothing reads. A typo
becomes a carried value rather than an error.

One of them is open by design: ``TrainingSettings.metadata`` accepts any key,
because configs carry free-form ones (``note``, ``group``, ``marker_source``,
...) that are prose for a human reader, and rejecting them would fail the
config-load gate for no gain. Its ``status`` field is closed, which is the half
that matters: a free-text status lets an arm admit an unimplemented mechanism in
words nothing reads.

Twenty-one classes reachable from a config are ``extra="allow"``:

* ``AdapterStepSchema``
* ``ColdParams`` [#du]_
* ``ExperimentMetadataSchema``
* ``KSDAuditConfigSchema``
* ``LatentDiffusionParams`` [#du]_
* ``LatentTrainingConfigSchema``
* ``TrainingConfigAcqHypernetwork`` [#md]_
* ``TrainingConfigBlochManifoldDPS`` [#md]_
* ``TrainingConfigCorruptionCalibration`` [#md]_
* ``TrainingConfigDataEfficiencyHarness`` [#md]_
* ``TrainingConfigDispersionBlochAE`` [#md]_
* ``TrainingConfigEquivarianceConformal`` [#md]_
* ``TrainingConfigIBVF`` [#md]_
* ``TrainingConfigPairedSynthesis`` [#md]_
* ``TrainingConfigPhysResidualConformal`` [#md]_
* ``TrainingConfigSE3Navigator`` [#md]_
* ``TrainingConfigTissueDiffusionPretrain`` [#md]_
* ``TrainingConfigTwinDPS`` [#md]_
* ``TrainingStrategyConfigSchema``
* ``TransformSpecSchema``
* ``UnspecifiedParams`` [#du]_

.. [#du] Reached only through the discriminated union at
   ``TrainingStrategyConfigSchema.diffusion``. A schema walk that unwraps a fixed
   number of ``get_args`` levels does not see these, which is why an earlier
   version of this page listed five.

.. [#md] A *top-level* schema, selected by ``training_mode`` rather than held as
   a field of anything. A walk seeded only from ``TrainingSettings`` never
   arrives at one, which is why a later version of this page listed eight. The
   dispatch table is ``_MODE_DISPATCH`` in
   ``spectramr/config/schemas/training/__init__.py``; a mode with no entry there
   loads through the default schema, which is already counted above.

These are deliberately open — they carry strategy- and transform-specific
payloads whose keys vary by component — so this is a documented trade-off rather
than a defect. Spell keys inside them carefully; nothing will tell you if you do
not. The three diffusion parameter blocks matter most in practice: an
``extra="allow"`` class is where a misspelled cold-diffusion or latent-diffusion
knob is carried silently instead of rejected.

``config/presets/`` cannot load its own presets
-----------------------------------------------

The ``spectramr.config.presets`` subsystem is importable and documented, and
resolves nothing. ``discover_and_register_presets()`` returns ``0`` and
``get_baseline_presets()`` returns ``[]``, because discovery defaults to an
experiments directory and globs ``experiment_*.yaml`` while the three shipped
presets are named ``baseline_*.yaml``. The example in
``spectramr/config/presets/cli.py``'s own docstring raises ``KeyError``.

Do not build on this module. Load configuration from a YAML file instead, which
is the path everything else uses.

Continuous integration runs a reduced lane
------------------------------------------

``.github/workflows/`` ships and executes: ``pr-required.yml`` fires on pull
requests, and its blocking jobs are ``lint-diff``, ``guards``, ``hygiene``,
``architecture``, ``unit-collect``, ``physics`` and ``security``, aggregated by
``required`` -- the single context branch protection asks for. That list is
pinned against the workflow file by ``tests/unit/ci/test_workflow_triggers.py``.

The lane is smaller than the framework's checks would allow, because some of them
have no subject here: a YAML-audit tier needs an experiment corpus this
distribution does not ship (see below), and a gate that cannot see its subject is
removed rather than left to report success.

**Pushing a version tag publishes.** ``release.yml`` triggers on ``push:`` of a
``v*`` tag and uploads to PyPI through Trusted Publishing. That is not a local,
reversible act, and PyPI never re-issues a filename it has already seen. Push the
commit first, confirm the lane is green, and tag only when you intend to release.

``make gate`` runs the same lane locally: it *derives* it by parsing
``pr-required.yml`` rather than restating it, so the two cannot drift.

.. code-block:: console

   $ make gate                                    # the whole lane
   $ make gate GATE_ARGS="--list"                 # what it derived, without running
   $ make gate GATE_ARGS="--jobs guards --quiet"  # one job

It reports three states and never folds the third into the first: ``PASS``,
``FAIL`` (exit 1) and ``UNRUNNABLE`` (exit 2, a required tool is absent). An
unrunnable step is not a pass -- which is the same distinction the next entry on
this page makes about a green ``audit``.

Auditing
========

A green ``audit`` does not mean every check ran
-----------------------------------------------

``spectramr audit`` runs about 150 config checks. On any given arm a number of them
do not run -- 17 of 144 on the arm sampled below. A check that declined to run is
reported with ``passed: true`` at ``severity: info``, does not affect the exit
code, and **is printed with the same green tick as a check that passed**:

.. code-block:: text

   ✅ [health:latent_decode_resolution] Not a latent model; skipping.
   ✅ [health:train_val_split_leakage] split-leakage check skipped (explicit_manifest):
      train and/or validation manifest not present locally.

Most of those are correct -- a latent-decode check has nothing to say about a
non-latent arm, and reporting it as a failure would be noise. Two groups are
structural rather than arm-shaped, and are worth knowing before you read a green
audit as coverage.

**Split-leakage is not checked in a clone.** ``train_val_split_leakage`` compares
the train and validation manifests. Those manifests are generated, not committed
-- ``data/manifests`` is in ``.gitignore`` -- so on an arm that declares
``index_path`` / ``validation_index_path`` the check has nothing to open and
skips. It was skipped on all five arms sampled across five cohorts: four for the
missing manifest, and one whose arm resolves its splits by directory crawl
instead, which the check reports separately as folder-disjoint. Generate the
manifests before you rely on this check.

**The five ``workflow_*`` checks need a ``workflow:`` block.** When an arm does
not declare one they skip, one for one -- confirmed against a sample where the
single arm carrying a ``workflow:`` block reported its regime and the other four
skipped. This is deliberate: an absent declaration is advisory and a wrong one is
a hard error, so the checks decline rather than guess. It does mean an arm with
no ``workflow:`` block is not being checked against the regime contract.

To see which checks declined on your own arm:

.. code-block:: console

   $ spectramr audit <arm>.yaml | grep -i skip

A tick beside the word *skipping* is a check that did not run. That is not the
same as a check that passed.

Optional dependencies
=====================

Mamba/SSM models require a separate, compiled install
-----------------------------------------------------

The ``.[mamba]`` extra is deliberately excluded from ``.[all]`` and ``.[dev]``:
it compiles a CUDA selective-scan kernel and needs ``nvcc`` plus a torch that is
already installed. Install it as a second step:

.. code-block:: console

   $ pip install -e '.[mamba]' --no-build-isolation

Without it, Mamba models **fail loudly** rather than degrading. The Gated-Conv +
GRU fallback exists only for CPU/CI wiring tests and is opt-in through
``SPECTRAMR_ALLOW_MAMBA_FALLBACK``; it is not an approximation of the Mamba path
and must not be used for results.

Some optional features have no ``pip install`` route
----------------------------------------------------

spectraMR declares seventeen optional-dependency extras, and the pattern is
consistent: a feature that needs a package outside the core install gets an
extra, and its runtime guard raises a clean ``ImportError`` naming that package.
``pip install "spectramr[iqa]"`` enables the closed-form perceptual metrics,
``[topology]`` enables the persistent-homology losses, and so on.

**Twenty optional packages are guarded in the source but appear in no extra.**
The guard half works — you get an accurate error naming the package — but there
is no ``spectramr[...]`` group that installs it, ``[all]`` included. You have to
install it yourself.

Four registered component names are affected. Each is present in its registry, so
a configuration may name it, and each raises at construction until you install
the package by hand:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25

   * - Kind
     - Registered name
     - Install first
   * - model
     - ``medical_dino_encoder``
     - ``pip install timm``
   * - loss
     - ``dino_perceptual``
     - ``pip install timm``
   * - metric
     - ``frd``
     - ``pip install pyradiomics SimpleITK``
   * - metric
     - ``rfs``
     - ``pip install pyradiomics SimpleITK``

Three more features degrade rather than fail, and say so in the log:

* ``latent_gan_generator`` raises only when ``text_conditioning_enabled`` is set
  (needs ``transformers``).
* ``symbolic_regression_wrapper`` trains normally but skips equation discovery
  without ``pysr``.
* ``neural_ode`` integrates with its built-in fixed-step RK4 without
  ``torchdiffeq``. Both solvers walk the same grid, so this changes the
  implementation and not the discretisation — but torchdiffeq's adjoint and
  adaptive solvers are unreachable.

The remaining undeclared packages guard non-registry code paths: ``beartype``,
``pydicom``, ``safetensors``, ``sigpy``, ``GPUtil``, ``ants``, ``fastrad``,
``flash_attn``, ``keras``, ``nvtx``, ``pypapi``, ``torch_fidelity``,
``torch_geometric`` and ``xformers``.

This is tracked upstream. Declaring the extras is the intended repair; it needs a
resolvable version bound checked against a package index for each one, and a
decision per package, because two of them (``flash_attn``, ``xformers``) compile
CUDA kernels and could not sit inside ``[all]`` any more than ``[mamba]`` can.


The published tree
==================

What the shipped package does not carry, and what that costs a reader who
expects it. Merged from the release-sanitization branch, which maintained this
page independently.

The experiment corpus is not published
--------------------------------------


The experiment corpus this framework was developed against is not part of this
release: those configurations encode one site's data layout and cluster
allocation, and several reference datasets whose licences do not permit
redistribution of derived data.

What ships is ``experiments/templates/comprehensive_config_template.yaml`` and
three baseline arms under ``experiments/inprogress/``. The template is a template
rather than an arm that was run: writing a new configuration against it is
possible; reading a known-good arm that produced a published result is not.

The other files under ``experiments/templates/`` are withheld on purpose rather
than overlooked: a template is *copied*, so one that does not parse is worse than
an absent one -- the reader's first act inherits the defect. What ships was
measured with ``spectramr audit``, not assumed.

Cluster and scheduler integration is unconfigured
-------------------------------------------------


The SLURM submission helpers generate batch scripts but ship with **no default
account, partition or mail address**. This is deliberate. A placeholder such as
``--account=your-account`` would be worse than an omission, because SLURM rejects an
unknown account outright, whereas an omitted ``#SBATCH --account`` directive lets
the scheduler apply the submitter's own default.

Supply your own values through ``ResourceSpec`` or the environment; see
``.env.example`` for the full set of variables and which component reads each one.

Some paradigm strategies read batch keys that no dataset produces
-----------------------------------------------------------------


The framework carries strategies for a wide range of paradigms, and several of the
more specialised ones read quantities from the training batch that no shipped data
pipeline emits -- ``T1_map``, ``brain_mask``, ``gradient_waveform``,
``resonance_params``, ``deformation_field`` and others.

A static census of the shipped source finds **59** such keys, of which 55 have a
read site in ``src/``. What happens when the key is
absent is not uniform, and the difference is the whole point:

* **23** raise. The absence is loud, the run stops, and the message names the key.
  This is the correct shape and needs no warning here.
* **32** are guarded, and the term they feed simply does not contribute. Training
  completes and reports success with one component of the objective silently
  inactive.

The second group is the reason this section exists. Such a strategy is not broken,
but it is **not exercised by any configuration that ships**: it needs a dataset that
emits those keys, which is a data-layer integration this release does not include.
Treat those paradigms as reference implementations to build against, not as paths
that run end to end on the shipped example arms.

At least one case is worse than an inactive term: the QSM pipeline substitutes an all-ones brain mask when none is supplied and runs
three physics operators on it, which produces a susceptibility map that is undefined
rather than approximate.

**The counts above are a static AST census and should be read as such.** The tool
cannot distinguish a training batch from any other local named ``batch``, which is
why it is a report and not a CI gate; and its own worst category -- a mechanism
running on a fabricated value -- reads ``0`` while at least one instance exists,
because it inspects only the default argument of ``.get()`` and not a fabrication in
the following statement.

Issue references in history and docstrings do not resolve
---------------------------------------------------------


This repository begins at a single initial commit. Rationale comments throughout
the source cite issue and pull-request numbers from the tracker the work was done
in -- roughly 386 distinct numbers across 585 files. Those numbers do not
correspond to issues here, and GitHub will autolink ``#N`` to whatever issue or PR
happens to hold that number in this repository.

They are kept rather than stripped because the surrounding sentence is usually the
only record of *why* a non-obvious piece of code is shaped the way it is, and a
rationale with a dangling reference is still a rationale. Read them as provenance,
not as links.

Running the test suite requires a git checkout
-----------------------------------------------


The suite enumerates the experiment corpus from ``git ls-files`` rather than a
filesystem glob, through the single owner ``tests/utils/corpus.py``. That choice
is deliberate and is documented in the module itself: a glob has a different
subject on every machine, and one cluster job once spent 24 of its failures on
two arms that exist in no git history at all.

The consequence is that **a tree with no** ``.git`` **cannot enumerate the
corpus**. That is not an error -- an sdist, a release tarball and a downloaded
ZIP are all legitimate ways to obtain this code, and none of them carry a
``.git``. In such a tree the corpus-dependent tests **skip visibly**, naming the
reason, and the rest of the suite runs normally:

.. code-block:: text

   SKIPPED [2] tests/utils/corpus.py:140: tests.utils.corpus enumerates
   git-tracked files, and <dir> sits in a tree with no .git -- an sdist,
   tarball or ZIP rather than a checkout.

To run those tests, obtain the code with ``git clone`` rather than as an archive.

A tree that *does* have a ``.git`` but whose git is unusable -- git missing from
``PATH``, a corrupt repository, a ``safe.directory`` refusal -- still raises,
loudly, and deliberately. Those two absences look identical from the outside and
call for opposite responses: the first has nothing to fix, while the second would
silently disable a large group of tests in a working tree if it were softened
into a skip.

Note that the published tree does not carry the experiment corpus at all (see
the first entry on this page), so most of these tests have no subject here even
in a clone.
