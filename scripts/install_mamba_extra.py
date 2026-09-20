#!/usr/bin/env python3
"""Install the ``mamba`` extra with CUDA kernels for the target cluster's GPUs.

``pip install -e '.[mamba]' --no-build-isolation`` -- the line every doc page
carried before this script -- cannot produce a kernel that runs on a V100, and
fails in a way indistinguishable from success. Four things have to be overridden,
and the one an experienced reader reaches for first is a no-op:

* **pip usually never runs the build.** An already-satisfied requirement or a
  cached wheel short-circuits it, and ``mamba-ssm`` / ``causal-conv1d`` each
  install a ``CachedWheelsCommand`` that downloads a prebuilt wheel before
  building. So: uninstall first, ``--no-cache-dir``, and
  ``MAMBA_FORCE_BUILD`` / ``CAUSAL_CONV1D_FORCE_BUILD``.
* **``TORCH_CUDA_ARCH_LIST`` is inert for these two.** Both hardcode their
  ``-gencode`` flags, and ``torch.utils.cpp_extension`` drops the flags it would
  derive from that variable once a user flag contains the substring ``arch``.
  ``NVCC_APPEND_FLAGS``, read by the **nvcc driver itself**, is the lever that
  a package's own gencode list cannot suppress.
* **Upstream's hardcoded list carries no ``sm_70``** at any toolkit version, so a
  V100 raises ``cudaErrorNoKernelImageForDevice`` at the first selective-scan
  launch -- the failure the cu126 torch pin exists to prevent, one layer down.

A forgotten ``FORCE_BUILD`` looks exactly like a successful build until a job
lands on a V100, so this ends by reading the arch list back out of the built
``.so`` with ``cuobjdump --list-elf``: what the artifact contains, never what the
flags should have produced.

Invoke it as ``make install-mamba`` (``ARCH_LIST=`` overrides the default
``7.0;8.9``, Volta plus Ada). ``docs/getting_started.rst`` carries the measured
arch lists, why Ada was never broken, why ``+PTX`` is omitted, and why
``flash-attn`` cannot be extended this way.

Exit codes: ``0`` built and every requested arch present; ``1`` the build or the
arch verification failed; ``2`` usage or toolchain error (no ``nvcc``, no
``ninja``, or a toolkit major disagreeing with torch's).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

DEFAULT_ARCH_LIST = "7.0;8.9"

# The two kernel extensions the `mamba` extra builds, paired with the env var
# that makes each package compile instead of fetching a prebuilt wheel.
KERNEL_MODULES = ("selective_scan_cuda", "causal_conv1d_cuda")
PACKAGES = ("mamba-ssm", "causal-conv1d")
FORCE_BUILD_VARS = ("MAMBA_FORCE_BUILD", "CAUSAL_CONV1D_FORCE_BUILD")

_ARCH_RE = re.compile(r"^\d+\.\d+$")
_SM_RE = re.compile(r"\bsm_(\d+)\b")

REPO_ROOT = Path(__file__).resolve().parents[1]


class ToolchainError(RuntimeError):
    """The environment cannot build or inspect a CUDA extension (exit 2)."""


class VerificationError(RuntimeError):
    """The built artifact does not carry the requested architectures (exit 1)."""


def parse_arch_list(spec: str) -> tuple[str, ...]:
    """Validate a ``7.0;8.9``-style spec into ordered, de-duplicated arches."""
    arches: list[str] = []
    for token in re.split(r"[;,\s]+", spec.strip()):
        if not token:
            continue
        if not _ARCH_RE.match(token):
            raise ValueError(
                f"{token!r} is not a compute capability; expected MAJOR.MINOR, e.g. 7.0"
            )
        if token not in arches:
            arches.append(token)
    if not arches:
        raise ValueError(f"no compute capability in {spec!r}")
    return tuple(arches)


def sm_token(arch: str) -> str:
    """``7.0`` -> ``sm_70``; ``10.0`` -> ``sm_100``."""
    major, minor = arch.split(".")
    return f"sm_{major}{minor}"


def gencode_flags(arches: Iterable[str]) -> list[str]:
    """One ``-gencode`` pair per arch, in nvcc's own argument form."""
    flags: list[str] = []
    for arch in arches:
        major, minor = arch.split(".")
        flags += ["-gencode", f"arch=compute_{major}{minor},code=sm_{major}{minor}"]
    return flags


def nvcc_release_version() -> str:
    """Return nvcc's release version (e.g. ``12.6``) or raise ``ToolchainError``."""
    try:
        proc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise ToolchainError(
            "nvcc is not on PATH. The mamba extra compiles a CUDA kernel, so it needs the "
            "toolkit, not just a CUDA-enabled torch wheel (on the clusters: module load cuda/12.6)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ToolchainError(f"`nvcc --version` failed: {exc.stderr}") from exc
    match = re.search(r"release (\d+\.\d+)", proc.stdout)
    if match is None:
        raise ToolchainError(f"could not read a release version out of:\n{proc.stdout}")
    return match.group(1)


def assert_toolchain() -> tuple[str, str]:
    """Return ``(nvcc release, torch's CUDA)``, refusing anything that cannot build.

    A major-version mismatch is refused rather than warned about: mamba's
    setup.py answers one by clearing ``TORCH_CUDA_ARCH_LIST``, so the build
    would succeed and produce a differently-targeted kernel.
    """
    nvcc_version = nvcc_release_version()
    try:
        import torch
    except ImportError as exc:
        raise ToolchainError(
            "torch is not importable. `--no-build-isolation` means the build imports it, "
            "so install the base environment first: pip install -e '.[all]'"
        ) from exc
    if torch.version.cuda is None:
        raise ToolchainError(
            f"torch {torch.__version__} is a CPU build; the mamba kernel has no CPU variant."
        )
    torch_version = str(torch.version.cuda)
    if nvcc_version.split(".")[0] != torch_version.split(".")[0]:
        raise ToolchainError(
            f"nvcc is CUDA {nvcc_version} but torch was built against {torch_version}."
        )
    # Asked of torch's own predicate, which shells out to the executable. Without
    # ninja, BuildExtension ignores MAX_JOBS and compiles serially, which turns a
    # ten-minute build into an hours-long one nobody lets finish -- and ninja is
    # declared in no dependency group, so it is absent on a fresh node.
    from torch.utils.cpp_extension import is_ninja_available

    if not is_ninja_available():
        raise ToolchainError(
            "ninja is not available, so the build would ignore MAX_JOBS and compile "
            "serially (hours, not minutes). Fix with: pip install ninja"
        )
    return nvcc_version, torch_version


def build_env(arches: Sequence[str], jobs: int) -> dict[str, str]:
    """The environment the build runs under, as a full copy with the deltas applied."""
    env = dict(os.environ)
    for var in FORCE_BUILD_VARS:
        env[var] = "TRUE"
    appended = " ".join(gencode_flags(arches))
    existing = env.get("NVCC_APPEND_FLAGS", "").strip()
    env["NVCC_APPEND_FLAGS"] = f"{existing} {appended}".strip()
    env["TORCH_CUDA_ARCH_LIST"] = ";".join(arches)
    env["MAX_JOBS"] = str(jobs)
    return env


def parse_cuobjdump_arches(text: str) -> set[str]:
    """Collect the ``sm_XX`` tokens out of ``cuobjdump --list-elf`` output."""
    return {f"sm_{digits}" for digits in _SM_RE.findall(text)}


def find_kernel_module(name: str) -> Path:
    """Locate an installed kernel extension by import machinery, not by globbing."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        raise VerificationError(
            f"{name} is not importable, so the extra did not install its CUDA kernel."
        )
    return Path(spec.origin)


def module_arches(so_path: Path) -> set[str]:
    """Read the architectures actually embedded in a built extension."""
    try:
        proc = subprocess.run(
            ["cuobjdump", "--list-elf", str(so_path)], capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        raise ToolchainError(
            "cuobjdump is not on PATH; it ships with the CUDA toolkit alongside nvcc."
        ) from exc
    if proc.returncode != 0:
        raise ToolchainError(f"cuobjdump failed on {so_path}: {proc.stderr.strip()}")
    return parse_cuobjdump_arches(proc.stdout)


def verify_arches(arches: Sequence[str]) -> list[str]:
    """Report each kernel's embedded arch list and return the shortfalls."""
    # The kernels were installed by the subprocess above, after this interpreter
    # cached the directory listing -- without this the verification reports a
    # good build as uninstalled, which would teach everyone to distrust it.
    importlib.invalidate_caches()
    wanted = {sm_token(arch) for arch in arches}
    shortfalls: list[str] = []
    for name in KERNEL_MODULES:
        so_path = find_kernel_module(name)
        present = module_arches(so_path)
        print(f"  {name}: {so_path.name} carries {', '.join(sorted(present)) or '(none)'}")
        missing = sorted(wanted - present)
        if missing:
            shortfalls.append(f"{name} is missing {', '.join(missing)}")
    return shortfalls


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = parser.add_argument
    add(
        "--arch-list",
        default=DEFAULT_ARCH_LIST,
        help=f"compute capabilities (default: {DEFAULT_ARCH_LIST})",
    )
    add(
        "--jobs",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help="MAX_JOBS; each nvcc job peaks at several GB",
    )
    add("--dry-run", action="store_true", help="print the env and command, build nothing")
    add("--verify-only", action="store_true", help="read the arch list off an existing install")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        arches = parse_arch_list(args.arch_list)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # pip short-circuits on an already-satisfied requirement and on a cached wheel,
    # and in both cases setup.py -- and with it every variable below -- never runs.
    # A node that already took the bare pip line is the common starting state.
    pip = [sys.executable, "-m", "pip"]
    uninstall = [*pip, "uninstall", "-y", *PACKAGES]
    command = [*pip, "install", "-e", ".[mamba]", "--no-build-isolation", "--no-cache-dir"]
    try:
        if args.verify_only:
            print(f"Verifying {', '.join(sm_token(a) for a in arches)} in the installed kernels:")
            shortfalls = verify_arches(arches)
        else:
            nvcc_version, torch_version = assert_toolchain()
            env = build_env(arches, args.jobs)
            print(f"arches      {';'.join(arches)}  ->  {', '.join(sm_token(a) for a in arches)}")
            print(f"nvcc        {nvcc_version} (torch built against {torch_version})")
            for var in (*FORCE_BUILD_VARS, "NVCC_APPEND_FLAGS", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS"):
                print(f"{var:<28}{env[var]}")
            print(f"uninstall   {' '.join(uninstall)}")
            print(f"command     {' '.join(command)}")
            if args.dry_run:
                return 0
            # Absent packages exit 0 with a warning, so non-zero means the old
            # wrong-arch build is still in place and pip will skip the build.
            if subprocess.run(uninstall, cwd=REPO_ROOT, env=env).returncode != 0:
                print("ERROR: could not remove the existing kernels.", file=sys.stderr)
                return 1
            built = subprocess.run(command, cwd=REPO_ROOT, env=env)
            if built.returncode != 0:
                print(
                    "ERROR: the build failed; the arch verification below would be meaningless.",
                    file=sys.stderr,
                )
                return 1
            print("Verifying the built kernels:")
            shortfalls = verify_arches(arches)
    except ToolchainError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except VerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if shortfalls:
        print("ERROR: the requested architectures are not in the artifact:", file=sys.stderr)
        for line in shortfalls:
            print(f"  {line}", file=sys.stderr)
        retry = (
            "Re-run this script without --verify-only"
            if args.verify_only
            else "Re-read the FORCE_BUILD lines above"
        )
        print(
            "Either a prebuilt wheel was installed, or a previous install was left in place "
            f"and pip skipped the build, or the gencode flags did not reach nvcc. {retry}.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: every requested architecture is present in both kernels ({';'.join(arches)}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
