Versioning and release branches
===============================

Every ``spectraMR`` release is numbered ``MAJOR.MINOR.BUILD`` -- three integers,
nothing else. The scheme is `PEP 440 <https://peps.python.org/pep-0440/>`_
compatible, which is what allows it onto PyPI at all, and it agrees with
`Semantic Versioning <https://semver.org/>`_ about what the three positions mean.

What the three positions mean
-----------------------------

``MAJOR``
   An incompatible change to something published as stable: the
   :doc:`scripting_api` surface, the meaning of a config key, or the definition
   of a metric. Numbers produced either side of a major bump are not required to
   be comparable, and what moved is recorded rather than left to be discovered.

``MINOR``
   New capability, added compatibly -- a paradigm, a model, a loss, a metric, or
   a config key whose default preserves existing behaviour.

``BUILD``
   Everything else: a fix, a documentation correction, a packaging change. This
   is the position that advances for a rebuild of the same feature set, and it
   is the one the ``nightly`` branch moves between releases.

The four branches
-----------------

The published repository carries four long-lived branches. Three of them are
defined by what they *ship*, and the version tells you whether you are looking at
a release (``X.Y.B``) or at a build (``X.Y.B.devN``) -- the ref tells you which
build. The fourth is defined by what it *documents*.

.. list-table::
   :header-rows: 1
   :widths: 14 22 64

   * - Branch
     - Version it carries
     - What it is
   * - ``main``
     - ``X.Y.B``
     - **Holds the release.** The tag ``vX.Y.B`` is cut from here, and that tag
       is what publishes to PyPI. A commit on ``main`` is a state that was
       released or is about to be, so ``main`` and the newest ``vX.Y.B`` tag
       carry the same tree. A snapshot that is not a release does not belong
       here; it belongs on ``dev``.
   * - ``dev``
     - ``X.Y.B.devN``
     - **Ships builds.** Integration, cut from ``main``, and the branch pull
       requests target -- and the only ref from which
       ``.github/workflows/dev-publish.yml`` will publish
       ``X.Y.B.dev<run number>`` to PyPI as a pre-release. A change on ``dev`` is
       installable as soon as a maintainer pushes a ``dev-build-*`` tag on its
       head, rather than at the next release.
   * - ``nightly``
     - ``X.Y.B.devN``
     - **The latest unstable version** -- the newest build, named by a ref so it
       can be fetched without reading a version number first. Its history is
       deliberately not stable: it is *moved* to a commit ``dev`` has already
       published, never merged into, so do not base work on it and do not open
       pull requests against it. It exists to be installed and tried.
   * - ``docs``
     - whatever the reviewed snapshot carried
     - **Documents the other three.** The ref Read the Docs builds ``latest``
       from, so what the site serves is decided by which snapshot's
       documentation was last reviewed, rather than by what ``main`` holds or by
       what ``dev`` last published. It carries a whole export snapshot, not a
       ``docs/`` directory. Like ``nightly`` it is *moved*, never merged into,
       and nothing installs from it.

``X.Y.B`` on ``dev`` and ``nightly`` is the version being worked *towards*, not
the last one released: while ``main`` sits at ``0.1.0``, those two carry
``0.1.1.devN``. ``docs`` reports no version of its own -- it repeats whatever the
snapshot it was moved to reported, ``X.Y.B.devN`` after a ``dev`` snapshot and
``X.Y.B`` after a release one, which is why the version does not tell you that
you are looking at it.

The ``docs`` branch
-------------------

Read the Docs maps one version to one ref, and before this branch existed it
served two: ``stable`` from the newest ``vX.Y.B`` tag, and ``latest`` from
``main``. With ``latest`` reading ``main``, the published documentation could
only describe the release -- a documentation correction could not be read until
the next release carried it, and restoring ``main`` to a release tree took the
corrections back off the site. Pointing ``latest`` at ``docs`` separates the
three concerns: ``stable`` follows the tag, the PyPI pre-release follows
``dev``, and the site follows the reviewed documentation.

One ref is one version, so this branch does not *serve* three versions. It
carries the single documentation set that *describes* all four of them -- the
page you are reading.

**It carries a whole snapshot rather than only** ``docs/``. A tree holding just
that directory builds without an error and without content. Sphinx imports the
package to render the API pages, so with no importable ``spectramr`` present, 77
of the 105 rendered ``spectramr.*`` signatures disappear and seven of the nine
pages carrying an autodoc directive lose their entire API section. The build
still exits 0: ``docs/conf.py`` suppresses ``autodoc.import_object``, so a
module that cannot be imported costs a missing section rather than a warning,
and ``fail_on_warning: true`` then has nothing to fail on. Those figures are
recorded in ``.readthedocs.yaml`` together with the environment that produced
them.

**What separates it from** ``dev`` **is the mover, not the contents.** Both
usually carry the same documentation. ``dev`` advances on every export snapshot,
whatever state that documentation is in; ``docs`` advances only once a
snapshot's documentation has been read. That difference is the entire reason the two are
separate rows rather than one row with two names (non-negotiable 17)::

   git push --force origin <reviewed-snapshot-sha>:docs

**It exists only in the published repository.** The research repository already
carries branches under ``refs/heads/docs/`` -- 40 of them on 2026-09-06 -- and
git cannot hold a reference at ``docs`` and references beneath ``docs/`` at the
same time. Documentation is authored there like everything else -- a feature
branch into ``dev`` -- and reaches this branch like everything else reaches the
published repository: through an export snapshot (non-negotiable 21).

Installing each one
-------------------

.. code-block:: bash

   pip install spectramr          # main -- the newest release
   pip install --pre spectramr    # dev  -- the newest build, release or not

``--pre`` is the whole opt-in, and it is not a flag anyone passes by accident.
PEP 440 orders ``0.1.3.dev7`` before ``0.1.3``, so a resolver without it never
reaches a dev build even when one is newer.

To pin the exact build a ``nightly`` commit names, read ``__version__`` from that
commit and install it by number::

   pip install "spectramr==0.1.3.dev7"

What publishes a dev build
--------------------------

``.github/workflows/dev-publish.yml`` in the **public** repository stamps
``X.Y.B.dev<github.run_number>`` into every file that states the version, builds
and verifies the distribution with ``build_dist.py``, and uploads it to PyPI.

**A** ``dev-build-*`` **tag starts it**, and that is a constraint rather than a
preference. ``test_workflow_triggers.py::test_push_triggers_are_tag_only`` admits
a ``push:`` trigger only when it is scoped to tags -- a branch push is autonomous
CI, and no allowlist entry can license one. ``schedule:`` and
``workflow_dispatch`` both register from the **default branch**: on the public
repository that is ``main``, which carries the release rather than this lane, so
a cron fires zero times and a dispatch returns 404 until a release export puts
the file there. A tag push instead runs the workflow file *at the tagged
commit*, so it works from the moment ``dev`` carries the file, and a person
pushes it -- the same posture as the release lane. ``workflow_dispatch`` is kept
and starts working once a release carries the file to ``main``.

The **commit** decides whether anything is uploaded, not the event or the ref: the
``build`` job compares the commit it built with public ``dev``'s head
(``git ls-remote origin refs/heads/dev``) and the ``pypi`` job runs only when the
two match. A tag on any other commit builds and verifies the distribution and
uploads nothing, which is how a change to this file is rehearsed; if ``dev``
moved after the tag was pushed, tag the new head.

.. warning::

   **Nothing moves** ``nightly`` **today.** Each of its commits was made by hand,
   and on 2026-09-06 the branch sat 10 commits behind ``main`` -- carrying the
   documentation set that ``main`` had already corrected. A branch whose name
   means "the newest build" and which is advanced only when someone remembers
   states something false for as long as they do not, and it states it
   invisibly: the ref still exists and still resolves.

   Until the move is automated -- one job on this lane, after ``pypi``, pushing
   the published commit onto ``nightly`` -- advance it by hand, immediately after
   tagging ``dev``'s head::

      git push origin dev
      t=dev-build-$(date -u +%Y%m%d-%H%M); git tag "$t" dev && git push origin "$t"
      git push --force origin dev:nightly   # nightly NAMES a build; it is moved, not merged

Three properties of that lane are load-bearing rather than incidental:

**The counter is the run number, and nothing else.** PyPI never re-issues a
filename, so two runs may not produce the same wheel name. ``github.run_number``
is monotonic per workflow and cannot repeat -- but it **restarts if the workflow
file is renamed**, so renaming that file burns the filenames it has already
published and requires a ``BUILD`` bump on ``dev`` in the same change.

**The lane refuses to run from a release version.** If ``dev`` carries ``0.1.3``
rather than ``0.1.3.devN``, a ``0.1.3.dev<n>`` wheel would sort *before* the
0.1.3 that has already shipped. The stamp step stops there and names
``bump_version.py nightly --apply`` as the way past it.

**The workflow ships to both repositories and guards itself.** The export
allowlist selects ``.github/`` wholesale, and the overlay under
``scripts/release/public_overlay/`` is replace-only -- it cannot add a path the
allowlist is not already shipping -- so a public-repo-only workflow file is not
available as an option. Every job therefore carries
``if: github.repository == 'adnaneGdihi/spectraMR'``. The private research
repository this tree is exported from also carries a ``dev`` branch, and it is
that repository's main working branch -- without the guard, a run there
would attempt a PyPI upload.

Trusted Publishing is pinned to a **workflow filename**, so ``release.yml``'s
publisher entry does not cover ``dev-publish.yml``. A second entry is required on
pypi.org, and creating it is a maintainer's action.

Why ``.devN`` and not ``+build``
--------------------------------

A pre-release must sort **before** the release it precedes, and it must be
publishable. PEP 440 gives ``0.1.1.dev4 < 0.1.1``, so an installer resolving
``spectramr`` never prefers a nightly over the stable release of the same
number, and ``pip install --pre`` is the explicit opt-in.

The obvious-looking alternative, a local version such as ``0.1.1+build4``, is
not usable here: PyPI **rejects local versions outright**, so a number that
reads naturally would make the artefact unpublishable -- and it would fail at
upload time, after the tag exists. The constraint is the index's, not a
preference.

Where the number lives
----------------------

``src/spectramr/__init__.py`` holds ``__version__``, and ``[tool.hatch.version]``
reads it -- so the package metadata and the wheel filename are derived, never
restated. Three other files state the version independently
(``CHANGELOG.md``, ``CITATION.cff``, and the git tag), because each is read by
something that cannot import the package.

``scripts/release/build_dist.py`` is the **sole comparator** of that set: it
reads all of them, and fails the build on any
disagreement, with ``--expect-version`` adding the tag as a fifth. Nothing else
compares them, and nothing else should.

Because those files are written by hand and compared only at release time,
moving the number is done with one command rather than three edits:

.. code-block:: bash

   python scripts/release/bump_version.py show        # what every file says now
   python scripts/release/bump_version.py minor       # dry run: print the plan
   python scripts/release/bump_version.py minor --apply

   python scripts/release/bump_version.py nightly --apply   # 0.1.0 -> 0.1.1.dev1
   python scripts/release/bump_version.py release --apply   # 0.1.1.dev4 -> 0.1.1

``release`` is the only mode that touches ``CHANGELOG.md``: it cuts the
accumulated ``[Unreleased]`` section into a dated heading for the version being
released and opens a fresh empty one. A ``nightly`` bump deliberately leaves the
changelog alone -- a dev build is not a release, and giving it a changelog
heading would make the file's release history unreadable.

Publishing order
----------------

The order below is forced by what each step can see, not by preference:

#. **Public tree** -- the export snapshot reaches the published repository.
#. **Zenodo** -- the deposit is dispatched. Zenodo cannot read a private
   repository, and the deposition is irreversible once published.
#. **Tag** -- ``vX.Y.B`` is pushed last, because it triggers the PyPI upload and
   the release notes have to quote a DOI that already exists.

Tagging as part of the publish push inverts that order and cannot be undone:
PyPI never re-issues a filename it has seen, so a wrong wheel costs a version
number rather than an amendment.
