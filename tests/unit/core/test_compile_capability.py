"""Tests for the compile-capability cross-check."""

from __future__ import annotations

import pytest

from spectramr.core.compile_capability import (
    CompileUnsupportedError,
    compile_backend_for,
    triton_available,
)


class TestCompileBackendFor:
    def test_cuda_with_triton_resolves(self):
        assert compile_backend_for("cuda", triton=True) == "inductor"

    def test_cuda_without_triton_raises(self):
        """Inductor cannot emit CUDA kernels without Triton, and a run that
        asked to be compiled must not quietly be eager."""
        with pytest.raises(CompileUnsupportedError, match="needs Triton"):
            compile_backend_for("cuda", triton=False)

    def test_cpu_does_not_need_triton(self):
        """Inductor on CPU uses its C++/OpenMP path, so the requirement is
        per-device rather than per-backend."""
        assert compile_backend_for("cpu", triton=False) == "inductor"

    def test_mps_does_not_need_triton(self):
        assert compile_backend_for("mps", triton=False) == "inductor"

    @pytest.mark.parametrize("device_type", ["xpu", "hpu", "tpu", "", "CUDA"])
    def test_an_unknown_device_type_raises(self, device_type):
        with pytest.raises(ValueError, match=r"No torch\.compile backend"):
            compile_backend_for(device_type, triton=True)

    def test_it_does_not_validate_the_backend_vocabulary(self):
        """That belongs to ``CompileConfigSchema._validate_backend``, which
        checks against ``torch._dynamo.list_backends()``. Two owners of one
        vocabulary is the failure non-negotiable 17 names, so this function
        takes no backend argument at all."""
        import inspect

        assert "backend" not in {
            name for name in inspect.signature(compile_backend_for).parameters if name != "triton"
        }


class TestTritonAvailable:
    def test_returns_a_bool(self):
        assert isinstance(triton_available(), bool)

    def test_it_is_the_one_owner(self):
        """``models/blocks/triton_scan.py`` held the original copy. It must
        delegate rather than keep a second answer that can drift."""
        import pathlib

        import spectramr.models.blocks.triton_scan as ts

        source = pathlib.Path(ts.__file__).read_text()
        assert "from spectramr.core.compile_capability import" in source
        assert ts.triton_available() == triton_available()

    def test_it_does_not_import_triton_to_answer(self):
        """``find_spec`` rather than ``import``: asking whether a package is
        installed should not pay for importing it."""
        import pathlib

        import spectramr.core.compile_capability as mod

        source = pathlib.Path(mod.__file__).read_text()
        body = source.split('"""', 2)[-1]
        assert "find_spec" in body
        assert "import triton" not in body
