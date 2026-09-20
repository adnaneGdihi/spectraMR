"""``resolve_cache_root`` must be idempotent under its own callers' side effects.

Issue #618. ``main.py`` calls it at import, then unconditionally sets
``TMPDIR = <resolved root>``; ``accelerator.initialize_accelerator`` calls it
again. The TMPDIR branch appends ``spectramr_cache`` to whatever TMPDIR holds, so
the second call returned ``$TMPDIR/spectramr_cache/spectramr_cache`` -- a different
root than the one main.py had already exported as ``TORCH_HOME`` /
``XDG_CACHE_HOME`` / ``CUDA_CACHE_CONFIG``. Visible in every cluster job log as
two ``Cache root from TMPDIR:`` lines whose paths disagree.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from spectramr.infrastructure.config.env_resolver import (
    _CACHE_ENV_LAYOUT,
    CacheDirectoryError,
    configure_cache_environment,
    resolve_cache_root,
)

_VARS = ("SPECTRAMR_CACHE_ROOT", "TMPDIR")

#: Every variable ``configure_cache_environment`` owns. Spelled out rather than
#: derived from ``_CACHE_ENV_LAYOUT`` so that dropping one from the layout is a
#: test failure and not a silently smaller assertion.
_CACHE_VARS = (
    "TMPDIR",
    "TORCH_HOME",
    "TORCH_METRICS_CACHE",
    "XDG_CACHE_HOME",
    "CUDA_CACHE_CONFIG",
    "CUDA_CACHE_PATH",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_second_call_returns_the_same_root_after_tmpdir_is_rewritten(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clean_env.setenv("TMPDIR", str(tmp_path))

    first = resolve_cache_root()
    assert first == tmp_path / "spectramr_cache"

    # Exactly what main.py:39 does immediately after the first call.
    clean_env.setenv("TMPDIR", str(first))

    second = resolve_cache_root()
    assert second == first, (
        f"cache root drifted on the second call: {first} -> {second}; the "
        "TMPDIR suffix was re-appended (#618)."
    )


def test_resolution_is_pinned_into_the_documented_override_var(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The resolved root is stamped so every later caller agrees on it."""
    import os

    clean_env.setenv("TMPDIR", str(tmp_path))
    root = resolve_cache_root()
    assert os.environ["SPECTRAMR_CACHE_ROOT"] == str(root)


def test_explicit_override_still_wins_and_is_not_suffixed(
    clean_env: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    explicit = tmp_path / "chosen"
    clean_env.setenv("SPECTRAMR_CACHE_ROOT", str(explicit))
    clean_env.setenv("TMPDIR", str(tmp_path / "ignored"))

    assert resolve_cache_root() == explicit
    assert resolve_cache_root() == explicit


def test_root_is_announced_once_per_resolution(
    clean_env: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Both calls log, but they must now agree on the path they report."""
    clean_env.setenv("TMPDIR", str(tmp_path))
    with caplog.at_level(
        logging.INFO, logger="spectramr.infrastructure.config.env_resolver"
    ):
        first = resolve_cache_root()
        clean_env.setenv("TMPDIR", str(first))
        resolve_cache_root()
    reported = [r.message for r in caplog.records if "Cache root" in r.message]
    assert len(reported) == 2
    assert all(str(first) in message for message in reported), reported


class TestConfigureCacheEnvironment:
    """The cache block is one function so that no entry point can get a subset.

    It was inline in ``main.py``, so ``spectramr train`` got six variables and
    ``torchrun -m spectramr.cli train-distributed`` -- which never imports
    ``main.py`` -- got the three ``initialize_accelerator`` happened to repeat.
    The two it left out were both caches DeepSpeed writes during its own import.
    """

    @pytest.fixture
    def clean_cache_env(self, monkeypatch: pytest.MonkeyPatch):
        for var in (*_VARS, *_CACHE_VARS):
            monkeypatch.delenv(var, raising=False)
        return monkeypatch

    def test_every_variable_is_set_under_the_root(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import os

        applied = configure_cache_environment(tmp_path)

        assert set(applied) == set(_CACHE_VARS), (
            "the layout no longer covers the documented set; a variable dropped "
            "here goes back to defaulting under $HOME"
        )
        for var in _CACHE_VARS:
            assert os.environ[var].startswith(str(tmp_path)), (
                f"{var}={os.environ[var]!r} does not follow cache_root"
            )

    def test_directories_are_created(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A denied root must fail here, naming the path, not inside a foreign import."""
        applied = configure_cache_environment(tmp_path)
        for var, value in applied.items():
            assert Path(value).is_dir(), f"{var} -> {value} was not created"

    def test_triton_is_the_one_variable_an_operator_can_override(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """DeepSpeed's own warning asks operators to set it for NFS reasons."""
        import os

        chosen = str(tmp_path / "operator-chose-this")
        clean_cache_env.setenv("TRITON_CACHE_DIR", chosen)

        configure_cache_environment(tmp_path)

        assert os.environ["TRITON_CACHE_DIR"] == chosen

    def test_the_others_follow_the_root_even_when_already_exported(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Anti-vacuity for the test above, and the case that matters on a cluster.

        A site profile exporting ``XDG_CACHE_HOME=$HOME/.cache`` would, under
        ``setdefault`` semantics, keep torch's JIT extension build root inside
        ``$HOME`` -- exactly the failure this block exists to prevent. The
        documented control point is ``SPECTRAMR_CACHE_ROOT``, so these are assigned.
        """
        import os

        for var in _CACHE_VARS:
            if var != "TRITON_CACHE_DIR":
                clean_cache_env.setenv(var, str(tmp_path / "site-profile-set-this"))

        configure_cache_environment(tmp_path)

        for var in _CACHE_VARS:
            if var == "TRITON_CACHE_DIR":
                continue
            assert "site-profile" not in os.environ[var], (
                f"{var} kept a pre-existing value; a cluster profile pointing it "
                "at $HOME would survive and DeepSpeed would write there"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # What an uncreatable directory says.
    #
    # The eager mkdir already named the path. What it did not say is who chose
    # the path, and a real 4-rank job died on exactly that ambiguity::
    #
    #     Command 'train-distributed' failed:
    #         [Errno 13] Permission denied: '/triton_cache'
    #
    # ``/triton_cache`` reads as "the framework put my cache at /", which is
    # impossible: the root had resolved (it logged the path) and five sibling
    # directories under it were created before Triton's turn. The value was an
    # inherited TRITON_CACHE_DIR kept by the setdefault carve-out. The two
    # assertions below are the two facts that were missing.
    #
    # A file standing in for an unwritable parent is used deliberately: chmod
    # 000 is a no-op for root, so the CI-as-root case would silently stop
    # testing anything. ``mkdir`` under a *file* is ENOTDIR for everyone.
    # ─────────────────────────────────────────────────────────────────────────

    def test_an_inherited_triton_dir_that_cannot_be_created_says_it_was_inherited(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("")
        chosen = str(blocker / "triton_cache")
        clean_cache_env.setenv("TRITON_CACHE_DIR", chosen)

        with pytest.raises(CacheDirectoryError) as excinfo:
            configure_cache_environment(tmp_path / "root")

        message = str(excinfo.value)
        assert chosen in message, "the failing path is gone from the message"
        assert "TRITON_CACHE_DIR" in message, (
            "the message does not name the variable that owns the path, so the "
            "operator cannot tell which knob to change"
        )
        assert "INHERITED" in message, (
            f"the message does not say the value came from the environment: {message}"
        )
        assert "unset" in message.lower(), "no remedy offered"

    def test_the_error_is_still_an_oserror(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Callers already catching ``OSError`` must keep catching this."""
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("")
        clean_cache_env.setenv("TRITON_CACHE_DIR", str(blocker / "triton_cache"))

        with pytest.raises(OSError):
            configure_cache_environment(tmp_path / "root")

    def test_a_framework_chosen_dir_that_cannot_be_created_points_at_the_root_knob(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Anti-vacuity for the test above: the other branch names a *different* knob.

        Nothing was inherited here, so telling the operator to unset a variable
        would be wrong -- ``SPECTRAMR_CACHE_ROOT`` is the only control point for
        the five members that are assigned rather than setdefault'd.
        """
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("")

        with pytest.raises(CacheDirectoryError) as excinfo:
            configure_cache_environment(blocker / "cache_root")

        message = str(excinfo.value)
        assert "SPECTRAMR_CACHE_ROOT" in message, message
        assert "INHERITED" not in message, (
            f"a framework-chosen path was reported as inherited: {message}"
        )

    def test_an_uncreatable_resolved_root_names_the_source_it_came_from(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``resolve_cache_root`` creates the root before the block runs.

        Its bare ``mkdir`` produced the same pathless-remedy failure as the loop,
        and there the ambiguity is worse: three different sources can produce the
        root, and two of them are not the variable you would think to unset.
        """
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("")
        clean_cache_env.setenv("TMPDIR", str(blocker))

        with pytest.raises(CacheDirectoryError) as excinfo:
            resolve_cache_root()

        message = str(excinfo.value)
        assert "TMPDIR" in message, (
            f"the root's failure does not say which source produced it: {message}"
        )
        assert "SPECTRAMR_CACHE_ROOT" in message, "no remedy offered"

    def test_the_variable_names_match_the_env_ssot(self) -> None:
        """``core.env`` is the registry of framework env-var names.

        The layout cannot import it -- ``main.py`` calls this module *before*
        ``import torch``, and ``core/__init__`` pulls torch -- so the literals are
        pinned against the SSOT here instead.
        """
        from spectramr.core import env as env_ssot

        for var, _subdir, _overwrite in _CACHE_ENV_LAYOUT:
            attr = var
            assert getattr(env_ssot, attr, None) == var, (
                f"{var} is not registered in core.env under the name {attr!r}"
            )


class TestCompileArtifactsFollowTheRoot:
    """Compiled artifacts are files, and they must land on the managed disk.

    Triton kernels, Inductor's generated modules and the PTX JIT cache are all
    written during a run. On a cluster the default locations are the wrong ones
    twice over: node-local ``/tmp`` is invisible to the next job (so every job
    recompiles) and often small, while ``$HOME`` is frequently NFS.
    """

    @pytest.fixture
    def clean_cache_env(self, monkeypatch: pytest.MonkeyPatch):
        for var in (*_VARS, *_CACHE_VARS):
            monkeypatch.delenv(var, raising=False)
        return monkeypatch

    def test_inductor_is_pinned_not_inherited_through_tmpdir(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The planted violation.

        Inductor derives its default from ``tempfile.gettempdir()``, which
        *looks* like it follows ``TMPDIR``. But ``gettempdir`` memoizes on first
        call: touch it before this function runs and the redirect is inert,
        while the env var still reads correctly. Drop
        ``TORCHINDUCTOR_CACHE_DIR`` from the layout and this turns red.
        """
        import os
        import tempfile

        tempfile.tempdir = None
        tempfile.gettempdir()  # freeze it at the system default, as an early import would
        assert tempfile.gettempdir() != str(tmp_path)

        configure_cache_environment(tmp_path)

        try:
            assert os.environ["TORCHINDUCTOR_CACHE_DIR"].startswith(str(tmp_path))
            torch_inductor = pytest.importorskip(
                "torch._inductor.runtime.cache_dir_utils"
            )
            assert torch_inductor.cache_dir().startswith(str(tmp_path)), (
                "Inductor resolved outside the cache root -- the TMPDIR coupling "
                "does not survive a memoized tempdir"
            )
        finally:
            tempfile.tempdir = None

    def test_the_ptx_cache_uses_the_nvidia_documented_spelling(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``CUDA_CACHE_PATH`` is the variable the driver reads;
        ``CUDA_CACHE_CONFIG`` beside it is not one of NVIDIA's three (#2178).
        Without this the cache stays in ``~/.nv/ComputeCache``."""
        import os

        configure_cache_environment(tmp_path)
        assert os.environ["CUDA_CACHE_PATH"].startswith(str(tmp_path))

    def test_every_compile_artifact_path_is_a_real_directory(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Created eagerly, so a denied path fails here naming the variable --
        not later, inside a third-party compile step."""
        configure_cache_environment(tmp_path)
        import os

        for var in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
            assert Path(os.environ[var]).is_dir(), f"{var} was not created"

    def test_the_artifact_paths_do_not_collide(
        self, clean_cache_env: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Three compilers writing into one directory would make the caches
        impossible to size, clear or diagnose separately."""
        import os

        configure_cache_environment(tmp_path)
        paths = {
            os.environ[var]
            for var in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
        }
        assert len(paths) == 2
