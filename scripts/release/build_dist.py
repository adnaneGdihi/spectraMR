#!/usr/bin/env python3
"""Build the PyPI distribution and refuse to hand back artefacts that lie.

``python -m build`` succeeding is not the property worth checking. **PyPI forbids
re-uploading a filename it has already seen**, so every defect below ships
permanently and costs a version number rather than an amend. Each check is chosen
for a failure that is *quiet* at build time:

* **completeness** -- the wheel payload must equal ``src/spectramr`` exactly; a
  ``[tool.hatch.build] exclude`` can take the config presets with it.
* **version agreement** -- four files state the version independently and nothing
  else compares them. ``--expect-version`` adds the git tag as a fifth, the one
  disagreement that cannot be repaired: ``v0.1.1`` at a tree declaring ``0.1.0``
  publishes ``spectramr-0.1.0`` and spends that filename forever.
* **no build contamination** -- a ``.pyc`` in the payload means the tree was
  measured before it was built.
* **no internal path in the sdist** -- hatchling ships everything not gitignored,
  and the ``sdist exclude`` list is a *denylist*, so it carries whatever nobody
  remembered. In the research checkout that is ``.agent``, ``CLAUDE.md``,
  ``TODO/``, ``paper/``, ``data/`` and the cluster scripts. Judged against the
  public allowlist, which is the export's own SSOT.
* **entry point resolves** -- a console script names its module by string, so a
  rename leaves it pointing at nothing and lint sees only a string.

Run from the repository root::

    python scripts/release/build_dist.py                          # build + verify
    python scripts/release/build_dist.py --check-only dist/       # verify existing
    python scripts/release/build_dist.py --expect-version v0.1.0  # also pin the tag
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

PACKAGE = "spectramr"
_WHEEL_VERSION = re.compile(rf"^{PACKAGE}-(?P<v>[^-]+)-py3-none-any\.whl$")
_TAG_REF = re.compile(r"^(?:refs/tags/)?v?(?P<v>\d+\.\d+[^\s]*)$")
_DEV_VERSION = re.compile(r"^(?P<release>\d+\.\d+\.\d+)\.dev\d+$")
_UNRELEASED_HEADING = re.compile(r"^##[ \t]*\[Unreleased\]", re.M | re.I)


def wheel_payload_paths(wheel: Path) -> set[str]:
    """Package-relative paths the wheel actually carries (dist-info excluded)."""
    prefix = f"{PACKAGE}/"
    with zipfile.ZipFile(wheel) as z:
        return {n[len(prefix) :] for n in z.namelist() if n.startswith(prefix)}


def package_source_paths(src_root: Path) -> set[str]:
    """Files under ``src/<package>/`` that a complete wheel must carry."""
    return {str(p.relative_to(src_root)) for p in src_root.rglob("*") if p.is_file()}


def completeness(in_wheel: set[str], on_disk: set[str]) -> tuple[list[str], list[str]]:
    """Return (missing-from-wheel, present-only-in-wheel), both sorted."""
    return sorted(on_disk - in_wheel), sorted(in_wheel - on_disk)


def contamination(paths: set[str]) -> list[str]:
    """Payload entries that prove the source tree was dirty when it was built."""
    return sorted(
        p for p in paths if p.endswith((".pyc", ".pyo")) or "__pycache__" in Path(p).parts
    )


def declared_version(init_text: str) -> str | None:
    """``__version__`` as written in the package ``__init__``."""
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_text, re.M)
    return m.group(1) if m else None


def changelog_version(text: str) -> str | None:
    """The newest released version heading, skipping ``[Unreleased]``."""
    for m in re.finditer(r"^##\s*\[([^\]]+)\]", text, re.M):
        tag = m.group(1)
        if tag.lower() != "unreleased":
            return tag
    return None


def changelog_disagreement(text: str, version: str) -> str | None:
    """Why ``CHANGELOG.md`` fails to state ``version``, or ``None`` if it states it.

    **A release and a dev build record themselves in different places**, so string
    equality against the newest heading is the wrong test for one of the two.
    ``bump_version.py nightly`` writes no heading at all -- a dev build is not a
    release, and a heading per build makes the file's release history unreadable --
    so a ``0.1.3.dev1`` tree correctly carries ``0.1.2`` as its newest heading, and
    comparing the two strings rejects a tree that is right. That rejection is what
    made ``nightly`` unbuildable while ``docs/versioning.rst`` advertised it as the
    branch that "exists to be installed and tried".

    What a dev build must instead show is the open ``[Unreleased]`` section it
    accumulates into, and that its own release has not already been cut: a
    ``0.1.3.dev1`` beside a dated ``[0.1.3]`` heading is a build of a version that
    already shipped, which is the one dev-series mistake that costs a filename.
    """
    newest = changelog_version(text)
    m = _DEV_VERSION.match(version)
    if m is None:
        if newest is None:
            return f"no released heading -- {version!r} needs '## [{version}] - <date>'"
        return None if newest == version else f"newest heading {newest!r} != {version!r}"
    if _UNRELEASED_HEADING.search(text) is None:
        return f"no '## [Unreleased]' section for the {version} series to accumulate into"
    if newest == m.group("release"):
        return (
            f"{m.group('release')!r} already has a dated heading, so {version!r} builds a "
            "version that has already been released -- 'bump_version.py nightly' opens the next"
        )
    return None


def citation_version(text: str) -> str | None:
    """``version:`` from CITATION.cff, read without a YAML dependency."""
    m = re.search(r'^version:\s*["\']?([^"\'\s]+)', text, re.M)
    return m.group(1) if m else None


def tag_version(ref: str) -> str | None:
    """Reduce a git ref to the version it claims, or ``None`` if it claims none.

    ``refs/tags/v0.1.0``, ``v0.1.0`` and ``0.1.0`` all mean ``0.1.0``; CI hands over
    whichever of the three the workflow author reached for, and comparing the raw ref
    against a version string fails on every one of them for the wrong reason. Anything
    that is not a version at all returns ``None``, which ``version_disagreements``
    reports as unreadable -- never as agreement.
    """
    m = _TAG_REF.match(ref.strip())
    return m.group("v") if m else None


def version_disagreements(
    reference: str | None, others: dict[str, str | None], changelog: str
) -> list[str]:
    """Every source in ``others`` that does not state ``reference``, unreadable included.

    ``reference`` is whichever source the caller holds authoritative -- the wheel
    filename during a build, ``src/spectramr/__init__.py`` when ``bump_version.py
    show`` reports a working tree. ``changelog`` is the raw ``CHANGELOG.md`` text,
    judged by :func:`changelog_disagreement` rather than by equality, because the
    two version shapes state themselves there differently.

    **One comparator, two callers** (non-negotiable 17). ``bump_version.py show``
    used to reimplement this as ``len(set(values)) != 1``, which cannot express the
    dev-build shape and so reported ``DISAGREEMENT`` on every correct dev tree.
    A second comparator does not announce itself: both return a plausible verdict,
    and the divergence surfaces as a wrong version on PyPI rather than as an error.
    """
    problems = [f"{k}: unreadable" for k, v in others.items() if v is None]
    if reference is None:
        return [*problems, "reference version: unreadable"]
    problems += [
        f"{k}: {v!r} != {reference!r}"
        for k, v in others.items()
        if v is not None and v != reference
    ]
    if (why := changelog_disagreement(changelog, reference)) is not None:
        problems.append(f"CHANGELOG.md: {why}")
    return problems


def entry_point_target(pyproject: dict) -> str | None:
    """The console script's ``module:attr``, or None if undeclared."""
    return pyproject.get("project", {}).get("scripts", {}).get(PACKAGE)


def entry_point_module_path(target: str) -> str:
    """``spectramr.cli.app:main`` -> ``cli/app.py`` (package-relative)."""
    module = target.split(":", 1)[0]
    parts = module.split(".")[1:]
    return "/".join(parts) + ".py"


def sdist_roots(sdist: Path) -> set[str]:
    """Top-level names inside the sdist, with its version prefix stripped."""
    roots: set[str] = set()
    with tarfile.open(sdist) as t:
        for m in t.getmembers():
            parts = m.name.split("/")
            if len(parts) > 1 and parts[1]:
                roots.add(parts[1])
    return roots


#: Backend-written, never read from the tree: no allowance can ever name it.
_SDIST_BUILD_ARTEFACTS = frozenset({"PKG-INFO"})


def unadmitted_roots(roots: set[str], allows: list[str]) -> list[str]:
    """Sdist roots no allowance could ever admit a file under.

    One-directional and weaker than the export's own decision: a root with *some*
    allowance is not proven clean, but one with **none** holds nothing shippable.
    """
    # First path segment of each allowance: the roots that can hold a shipped file.
    patterns = {a.split("/", 1)[0] for a in allows}
    return sorted(
        r
        for r in roots - _SDIST_BUILD_ARTEFACTS
        if not any(fnmatch.fnmatch(r, p) for p in patterns)
    )


def load_allowances(repo: Path) -> list[str]:
    """Read the public allowlist through the export script's own parser.

    Not a reimplementation: ``!`` denials, comment stripping and the refusal of an
    allowance-free file are export_public_tree.py's rules, and a second copy would
    drift silently (#17). By path, since ``scripts/`` is not importable from here.
    """
    exporter = repo / "scripts" / "release" / "export_public_tree.py"
    allowlist = repo / "scripts" / "release" / "public_allowlist.txt"
    if not exporter.is_file() or not allowlist.is_file():
        raise FileNotFoundError(
            f"cannot judge the sdist: {exporter.name} or {allowlist.name} is absent"
        )
    spec = importlib.util.spec_from_file_location("_export_public_tree", exporter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    allows, _denies = module.parse_allowlist(allowlist.read_text(encoding="utf-8"), str(allowlist))
    return allows


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def _find(dist: Path, suffix: str) -> Path:
    hits = sorted(dist.glob(f"{PACKAGE}-*{suffix}"))
    if len(hits) != 1:
        raise SystemExit(
            f"expected exactly one {suffix} in {dist}, found {len(hits)}: "
            f"{[h.name for h in hits]} -- a stale artefact from a previous version "
            f"would otherwise be checked, or published, alongside the new one"
        )
    return hits[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument(
        "--check-only", type=Path, default=None, help="verify an existing dist/ instead of building"
    )
    ap.add_argument(
        "--expect-version",
        default=None,
        metavar="REF",
        help="a git tag or ref the artefacts must match "
        "(accepts refs/tags/v1.2.3, v1.2.3 or 1.2.3)",
    )
    args = ap.parse_args(argv)
    repo: Path = args.repo.resolve()
    dist = (args.check_only or repo / "dist").resolve()

    if args.check_only is None:
        for stale in dist.glob(f"{PACKAGE}-*"):
            stale.unlink()
        print("building ...")
        if _run([sys.executable, "-m", "build"], repo) != 0:
            return 1

    wheel, sdist = _find(dist, ".whl"), _find(dist, ".tar.gz")
    print(f"\nwheel : {wheel.name}  ({wheel.stat().st_size:,} bytes)")
    print(f"sdist : {sdist.name}  ({sdist.stat().st_size:,} bytes)")

    failures: list[str] = []
    payload = wheel_payload_paths(wheel)
    on_disk = package_source_paths(repo / "src" / PACKAGE)

    missing, extra = completeness(payload, on_disk)
    print(f"\ncompleteness : {len(payload)} in wheel / {len(on_disk)} on disk")
    for label, rows in (("missing from wheel", missing), ("only in wheel", extra)):
        for r in rows[:20]:
            failures.append(f"{label}: {r}")

    dirty = contamination(payload)
    print(f"contamination: {len(dirty)} build artefact(s) in the payload")
    failures += [f"build artefact in payload: {d}" for d in dirty[:20]]

    wheel_v = m.group("v") if (m := _WHEEL_VERSION.match(wheel.name)) else None
    changelog_text = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    others = {
        "__init__.py": declared_version(
            (repo / "src" / PACKAGE / "__init__.py").read_text(encoding="utf-8")
        ),
        "CITATION.cff": citation_version((repo / "CITATION.cff").read_text(encoding="utf-8")),
    }
    if args.expect_version is not None:
        others["git tag"] = tag_version(args.expect_version)
    shown = ", ".join(f"{k}={v}" for k, v in others.items())
    print(
        f"\nversions     : wheel={wheel_v}, {shown}, "
        f"CHANGELOG.md newest heading={changelog_version(changelog_text)}"
    )
    failures += [
        f"version disagreement -- {p}"
        for p in version_disagreements(wheel_v, others, changelog_text)
    ]

    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    target = entry_point_target(pyproject)
    if target is None:
        failures.append(f"no [project.scripts] entry for {PACKAGE!r}")
    else:
        rel = entry_point_module_path(target)
        ok = rel in payload
        print(f"entry point  : {target} -> {rel} {'OK' if ok else 'MISSING'}")
        if not ok:
            failures.append(f"console script {target!r} names {rel}, absent from the wheel")

    if "py.typed" not in payload:
        failures.append("py.typed absent, but the package advertises 'Typing :: Typed'")

    roots = sdist_roots(sdist)
    print(f"sdist roots  : {', '.join(sorted(roots))}")
    try:
        leaks = unadmitted_roots(roots, load_allowances(repo))
    except (FileNotFoundError, SystemExit) as exc:
        # Never a silent pass: unverifiable is a failure, not agreement.
        failures.append(f"sdist not verifiable against the public allowlist -- {exc}")
    else:
        print(f"sdist leaks  : {len(leaks)} root(s) the public allowlist does not admit")
        failures += [f"internal path in sdist: {r}" for r in leaks[:20]]

    print("\nrunning twine check ...")
    if _run([sys.executable, "-m", "twine", "check", str(wheel), str(sdist)], repo) != 0:
        failures.append("twine check failed -- PyPI would reject the metadata")

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for f in failures:
            print(f"  ! {f}")
        return 1
    print("\nOK -- artefacts are complete, consistent and publishable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
