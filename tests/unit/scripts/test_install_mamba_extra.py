"""Tests for scripts/install_mamba_extra.py.

The script's claim is narrow and load-bearing: after it runs, both CUDA kernels
of the ``mamba`` extra carry a cubin for **every** requested compute capability.
The failure it exists to catch is silent -- a forgotten ``*_FORCE_BUILD`` makes
pip install upstream's prebuilt wheel, whose arch list has no ``sm_70``, and the
install reports success until a job lands on a V100.

So the arch check is a detector, and every violation shape below is planted
rather than reasoned about (non-negotiable 15): a wheel carrying upstream's list
but not Volta, a partially-honoured list, an object with no cubins at all, an
uninstalled kernel, and a missing or failing ``cuobjdump``. The last two matter
because a detector that cannot run must not read as a detector that passed.

The module is loaded by file path (it lives under ``scripts/``, outside the
import package) following the same convention as the other script tests.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "install_mamba_extra.py"


def _load():
    spec = importlib.util.spec_from_file_location("install_mamba_extra", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ime = _load()

# What `cuobjdump --list-elf` prints for a wheel built from upstream's hardcoded
# gencode list: Turing through Hopper, and no Volta.
UPSTREAM_ELF = """
ELF file    1: selective_scan_cuda.1.sm_75.cubin
ELF file    2: selective_scan_cuda.2.sm_80.cubin
ELF file    3: selective_scan_cuda.3.sm_87.cubin
ELF file    4: selective_scan_cuda.4.sm_90.cubin
"""

VOLTA_AND_ADA_ELF = """
ELF file    1: selective_scan_cuda.1.sm_70.cubin
ELF file    2: selective_scan_cuda.2.sm_75.cubin
ELF file    3: selective_scan_cuda.3.sm_80.cubin
ELF file    4: selective_scan_cuda.4.sm_89.cubin
"""


# --------------------------------------------------------------------------
# The default is the cluster's hardware, and a later edit must not quietly
# narrow it -- this is the regression guard for the whole change.
# --------------------------------------------------------------------------


def test_default_arch_list_covers_volta_and_ada() -> None:
    arches = ime.parse_arch_list(ime.DEFAULT_ARCH_LIST)
    assert "7.0" in arches, "Volta (V100) dropped from the default arch list"
    assert "8.9" in arches, "Ada dropped from the default arch list"


def test_gencode_flags_are_nvcc_argument_pairs() -> None:
    assert ime.gencode_flags(("7.0", "8.9")) == [
        "-gencode",
        "arch=compute_70,code=sm_70",
        "-gencode",
        "arch=compute_89,code=sm_89",
    ]


@pytest.mark.parametrize(
    ("arch", "token"), [("7.0", "sm_70"), ("8.9", "sm_89"), ("10.0", "sm_100")]
)
def test_sm_token(arch: str, token: str) -> None:
    assert ime.sm_token(arch) == token


# --------------------------------------------------------------------------
# Arch-list parsing raises on every malformed shape rather than defaulting
# (non-negotiable 3).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["volta", "7", "sm_70", "7.x", "", "   ", ";;"])
def test_parse_arch_list_raises_on_garbage(spec: str) -> None:
    with pytest.raises(ValueError):
        ime.parse_arch_list(spec)


def test_parse_arch_list_dedupes_and_keeps_order() -> None:
    assert ime.parse_arch_list("8.9;7.0, 8.9 9.0") == ("8.9", "7.0", "9.0")


# --------------------------------------------------------------------------
# The build environment carries all three overrides, and does not clobber
# flags the caller already set.
# --------------------------------------------------------------------------


def test_build_env_forces_a_source_build_for_both_packages(monkeypatch) -> None:
    monkeypatch.delenv("NVCC_APPEND_FLAGS", raising=False)
    env = ime.build_env(("7.0", "8.9"), jobs=4)
    assert env["MAMBA_FORCE_BUILD"] == "TRUE"
    assert env["CAUSAL_CONV1D_FORCE_BUILD"] == "TRUE"


def test_build_env_appends_gencode_for_every_arch(monkeypatch) -> None:
    monkeypatch.delenv("NVCC_APPEND_FLAGS", raising=False)
    env = ime.build_env(("7.0", "8.9"), jobs=4)
    assert "arch=compute_70,code=sm_70" in env["NVCC_APPEND_FLAGS"]
    assert "arch=compute_89,code=sm_89" in env["NVCC_APPEND_FLAGS"]
    assert env["TORCH_CUDA_ARCH_LIST"] == "7.0;8.9"
    assert env["MAX_JOBS"] == "4"


def test_build_env_preserves_preexisting_nvcc_flags(monkeypatch) -> None:
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "-lineinfo")
    env = ime.build_env(("7.0",), jobs=1)
    assert env["NVCC_APPEND_FLAGS"].startswith("-lineinfo ")
    assert "arch=compute_70,code=sm_70" in env["NVCC_APPEND_FLAGS"]


# --------------------------------------------------------------------------
# Planted violations: each shape the arch check must turn red on.
# --------------------------------------------------------------------------


def test_parse_cuobjdump_reads_the_real_listing_format() -> None:
    assert ime.parse_cuobjdump_arches(UPSTREAM_ELF) == {"sm_75", "sm_80", "sm_87", "sm_90"}
    assert ime.parse_cuobjdump_arches("") == set()


def _stub_verify(monkeypatch, listing: str) -> None:
    monkeypatch.setattr(ime, "find_kernel_module", lambda name: Path(f"/fake/{name}.so"))
    monkeypatch.setattr(ime, "module_arches", lambda so: ime.parse_cuobjdump_arches(listing))


def test_upstream_wheel_is_reported_as_missing_volta(monkeypatch) -> None:
    _stub_verify(monkeypatch, UPSTREAM_ELF)
    shortfalls = ime.verify_arches(("7.0", "8.9"))
    assert len(shortfalls) == len(ime.KERNEL_MODULES)
    assert all("sm_70" in line and "sm_89" in line for line in shortfalls)


def test_partially_honoured_arch_list_names_only_the_gap(monkeypatch) -> None:
    _stub_verify(monkeypatch, VOLTA_AND_ADA_ELF)
    assert ime.verify_arches(("7.0", "8.9")) == []
    shortfalls = ime.verify_arches(("7.0", "8.9", "9.0"))
    assert all(line.endswith("is missing sm_90") for line in shortfalls)


def test_verify_invalidates_the_import_cache_first(monkeypatch) -> None:
    """A good build must not report as uninstalled.

    The kernels land in site-packages after this interpreter cached that
    directory's listing, so ``find_spec`` misses them unless the cache is
    dropped -- a false red here is worse than no check, because it is the shape
    that teaches a reader to ignore the gate.
    """
    calls: list[int] = []
    monkeypatch.setattr(importlib, "invalidate_caches", lambda: calls.append(1))
    _stub_verify(monkeypatch, VOLTA_AND_ADA_ELF)
    ime.verify_arches(("7.0", "8.9"))
    assert calls, "verify_arches did not invalidate the import cache"


def test_object_with_no_cubins_at_all_fails(monkeypatch) -> None:
    _stub_verify(monkeypatch, "")
    assert ime.verify_arches(("7.0",)) != []


def test_uninstalled_kernel_raises_rather_than_passing(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ime.VerificationError, match="did not install"):
        ime.find_kernel_module("selective_scan_cuda")


def test_missing_cuobjdump_is_a_toolchain_error_not_a_pass(monkeypatch) -> None:
    def _absent(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(ime.subprocess, "run", _absent)
    with pytest.raises(ime.ToolchainError, match="cuobjdump"):
        ime.module_arches(Path("/fake/selective_scan_cuda.so"))


def test_failing_cuobjdump_is_a_toolchain_error_not_a_pass(monkeypatch) -> None:
    monkeypatch.setattr(
        ime.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="not an ELF"),
    )
    with pytest.raises(ime.ToolchainError, match="not an ELF"):
        ime.module_arches(Path("/fake/selective_scan_cuda.so"))


# --------------------------------------------------------------------------
# Toolchain preconditions.
# --------------------------------------------------------------------------


def _fake_torch(monkeypatch, cuda: str | None, *, ninja: bool = True) -> None:
    """Install a torch stand-in, so the checks do not depend on the real install.

    The whole ``torch.utils.cpp_extension`` chain has to be present: the function
    under test reaches ``is_ninja_available`` through a from-import, which walks
    the parent packages unless every level is already in ``sys.modules``.
    """
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(cuda=cuda)  # type: ignore[attr-defined]
    torch.__version__ = "2.13.0"  # type: ignore[attr-defined]
    ext = types.ModuleType("torch.utils.cpp_extension")
    ext.is_ninja_available = lambda: ninja  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", types.ModuleType("torch.utils"))
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", ext)


def test_toolkit_major_mismatch_raises(monkeypatch) -> None:
    monkeypatch.setattr(ime, "nvcc_release_version", lambda: "13.0")
    _fake_torch(monkeypatch, "12.6")
    with pytest.raises(ime.ToolchainError, match=r"13\.0"):
        ime.assert_toolchain()


def test_cpu_only_torch_raises(monkeypatch) -> None:
    monkeypatch.setattr(ime, "nvcc_release_version", lambda: "12.6")
    _fake_torch(monkeypatch, None)
    with pytest.raises(ime.ToolchainError, match="CPU build"):
        ime.assert_toolchain()


def test_absent_ninja_raises_rather_than_compiling_serially(monkeypatch) -> None:
    """ninja is declared in no dependency group, so a fresh node may not have it.

    Without it BuildExtension ignores MAX_JOBS and compiles serially, which is a
    correct build nobody waits for — and a killed build leaves the partial
    install that makes pip skip the next attempt.
    """
    monkeypatch.setattr(ime, "nvcc_release_version", lambda: "12.6")
    _fake_torch(monkeypatch, "12.6", ninja=False)
    with pytest.raises(ime.ToolchainError, match="ninja"):
        ime.assert_toolchain()


def test_matching_toolkit_is_accepted(monkeypatch) -> None:
    monkeypatch.setattr(ime, "nvcc_release_version", lambda: "12.6")
    _fake_torch(monkeypatch, "12.6")
    assert ime.assert_toolchain() == ("12.6", "12.6")


def test_absent_nvcc_names_the_module_load(monkeypatch) -> None:
    def _absent(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(ime.subprocess, "run", _absent)
    with pytest.raises(ime.ToolchainError, match="module load cuda"):
        ime.nvcc_release_version()


# --------------------------------------------------------------------------
# Exit codes, end to end.
# --------------------------------------------------------------------------


def test_main_rejects_a_malformed_arch_list() -> None:
    assert ime.main(["--arch-list", "volta", "--dry-run"]) == 2


def test_main_dry_run_prints_the_env_and_builds_nothing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(ime, "assert_toolchain", lambda: ("12.6", "12.6"))
    monkeypatch.setattr(ime.subprocess, "run", lambda *a, **k: pytest.fail("built on a dry run"))
    assert ime.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "arch=compute_70,code=sm_70" in out
    assert "MAMBA_FORCE_BUILD" in out


def _spy_subprocess(monkeypatch, *, uninstall_rc: int = 0, install_rc: int = 0):
    """Record every argv `main` runs, so call ORDER is assertable and nothing builds."""
    calls: list[list[str]] = []

    def _run(argv, **kwargs):
        calls.append(list(argv))
        rc = uninstall_rc if "uninstall" in argv else install_rc
        return types.SimpleNamespace(returncode=rc, stdout="", stderr="")

    monkeypatch.setattr(ime, "assert_toolchain", lambda: ("12.6", "12.6"))
    monkeypatch.setattr(ime.subprocess, "run", _run)
    return calls


def test_the_existing_install_is_removed_before_building(monkeypatch) -> None:
    """pip skips the build when the requirement is already satisfied.

    A cluster node that already ran the bare pip line is the *common* starting
    state, and there the FORCE_BUILD vars and the gencode flags are never
    consulted at all — setup.py does not run. So the uninstall has to come
    first, and the wheel cache has to be refused.
    """
    calls = _spy_subprocess(monkeypatch)
    _stub_verify(monkeypatch, VOLTA_AND_ADA_ELF)
    assert ime.main([]) == 0
    assert len(calls) == 2, calls
    assert "uninstall" in calls[0] and "install" in calls[1]
    for package in ime.PACKAGES:
        assert package in calls[0]
    assert "--no-cache-dir" in calls[1]
    assert "--no-build-isolation" in calls[1]


def test_a_failed_uninstall_aborts_instead_of_building_over_it(monkeypatch) -> None:
    calls = _spy_subprocess(monkeypatch, uninstall_rc=1)
    _stub_verify(monkeypatch, VOLTA_AND_ADA_ELF)
    assert ime.main([]) == 1
    assert len(calls) == 1, "the install ran on top of a kernel that could not be removed"


def test_dry_run_shows_both_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(ime, "assert_toolchain", lambda: ("12.6", "12.6"))
    monkeypatch.setattr(ime.subprocess, "run", lambda *a, **k: pytest.fail("ran on a dry run"))
    assert ime.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "uninstall -y" in out
    assert "--no-cache-dir" in out


def test_main_verify_only_fails_on_an_upstream_wheel(monkeypatch) -> None:
    _stub_verify(monkeypatch, UPSTREAM_ELF)
    assert ime.main(["--verify-only"]) == 1


def test_main_verify_only_passes_on_a_volta_and_ada_build(monkeypatch) -> None:
    _stub_verify(monkeypatch, VOLTA_AND_ADA_ELF)
    assert ime.main(["--verify-only"]) == 0
