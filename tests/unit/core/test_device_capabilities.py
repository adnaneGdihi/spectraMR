"""Tests for the device-capability SSOT.

Every test here injects a capability rather than probing hardware: CI has no
GPU, and ``tests/conftest.py`` installs a torch ``MagicMock``, so a test that
probed would be asserting against a mock.
"""

from __future__ import annotations

import dataclasses

import pytest

from spectramr.core.device_capabilities import (
    MIN_NATIVE_BF16_CAPABILITY,
    TARGET_CAPABILITY_ENV,
    DeviceCapabilities,
    build_capabilities,
    native_bf16_supported,
    parse_compute_capability,
    supported_amp_dtypes,
    target_capability_from_env,
)


class TestNativeBf16:
    """The reason this module exists.

    ``torch.cuda.is_bf16_supported()`` defaults to ``including_emulation=True``
    and answers **True** on sm_70/sm_75, because that branch only checks a bf16
    tensor can be created. The target clusters are V100s. So the threshold has
    to be the capability, not the probe.
    """

    @pytest.mark.parametrize("capability", [(7, 0), (7, 5), (6, 1)])
    def test_pre_ampere_has_no_native_bf16(self, capability):
        assert native_bf16_supported(capability) is False

    @pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (9, 0)])
    def test_ampere_and_later_do(self, capability):
        assert native_bf16_supported(capability) is True

    def test_the_threshold_is_sm80(self):
        assert MIN_NATIVE_BF16_CAPABILITY == (8, 0)

    def test_unknown_is_not_support(self):
        """``None`` degrades toward the safe dtype, never the fast one."""
        assert native_bf16_supported(None) is False


class TestSupportedAmpDtypes:
    def test_pre_ampere_offers_no_bfloat16(self):
        assert supported_amp_dtypes((7, 0)) == ("float32", "float16")

    def test_ampere_adds_bfloat16(self):
        assert supported_amp_dtypes((8, 0)) == ("float32", "float16", "bfloat16")

    def test_float32_is_always_present(self):
        """It denotes the absence of autocast, not a hardware feature."""
        for capability in [(7, 0), (8, 9), None]:
            assert "float32" in supported_amp_dtypes(capability)

    def test_the_spellings_match_the_schema(self):
        """These strings are compared against ``optimization.precision.dtype``."""
        from spectramr.infrastructure.training.mixed_precision import (
            _AMP_DTYPE_TO_PRECISION,
        )

        assert set(supported_amp_dtypes((8, 9))) <= set(_AMP_DTYPE_TO_PRECISION)


class TestParseComputeCapability:
    def test_parses_a_dotted_pair(self):
        assert parse_compute_capability("8.9") == (8, 9)
        assert parse_compute_capability(" 7.0 ") == (7, 0)

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_unset_is_none(self, raw):
        assert parse_compute_capability(raw) is None

    @pytest.mark.parametrize("raw", ["sm_70", "8", "8.x", "seven", "8.0.1", "-8.0"])
    def test_garbage_raises_rather_than_defaulting(self, raw):
        """A knob that silently accepts nonsense is indistinguishable from one
        that works (pitfall #15)."""
        with pytest.raises(ValueError, match="not a compute capability"):
            parse_compute_capability(raw)

    def test_env_reader_delegates(self):
        assert target_capability_from_env({TARGET_CAPABILITY_ENV: "9.0"}) == (9, 0)
        assert target_capability_from_env({}) is None

    def test_the_env_name_has_one_owner(self):
        """Aliased from the registry, not respelled here."""
        from spectramr.core import env_names

        assert TARGET_CAPABILITY_ENV == env_names.SPECTRAMR_TARGET_COMPUTE_CAPABILITY
        assert "SPECTRAMR_TARGET_COMPUTE_CAPABILITY" in env_names.__all__


class TestBuildCapabilities:
    def test_a_v100_record(self):
        caps = build_capabilities("cuda", (7, 0), triton=True, source="test")
        assert caps.native_bf16 is False
        assert caps.compile_backend == "inductor"
        assert caps.incomplete == ()

    def test_an_unreadable_capability_is_reported_not_guessed(self):
        """``incomplete`` is what separates "no" from "cannot tell"."""
        caps = build_capabilities("cuda", None, triton=True, source="unknown")
        assert "capability" in caps.incomplete
        assert caps.native_bf16 is False

    def test_cuda_without_triton_cannot_name_a_backend(self):
        caps = build_capabilities("cuda", (8, 9), triton=False, source="test")
        assert caps.compile_backend is None
        assert "compile_backend" in caps.incomplete

    def test_cpu_needs_no_triton(self):
        """Inductor on CPU uses its C++ path, so the requirement is per-device."""
        caps = build_capabilities("cpu", None, triton=False, source="test")
        assert caps.compile_backend == "inductor"
        assert "compile_backend" not in caps.incomplete

    def test_cpu_missing_capability_is_not_incomplete(self):
        """A CPU has no compute capability; that is a fact, not a gap."""
        caps = build_capabilities("cpu", None, triton=False, source="test")
        assert caps.incomplete == ()


class TestSerialisation:
    def test_capability_renders_as_the_dotted_string(self):
        """``provenance.torch_runtime`` already spells it ``f"{major}.{minor}"``;
        one artifact must not carry two spellings."""
        caps = build_capabilities("cuda", (8, 9), triton=True, source="test")
        assert caps.capability_str == "8.9"
        assert caps.to_dict()["capability"] == "8.9"

    def test_unknown_capability_serialises_as_null(self):
        caps = build_capabilities("cuda", None, triton=True, source="unknown")
        assert caps.to_dict()["capability"] is None

    def test_the_record_is_frozen(self):
        caps = build_capabilities("cuda", (8, 9), triton=True, source="test")
        with pytest.raises(dataclasses.FrozenInstanceError):
            caps.native_bf16 = True  # type: ignore[misc]

    def test_to_dict_is_json_serialisable(self):
        import json

        caps = build_capabilities("cuda", (8, 9), triton=True, source="test")
        assert json.loads(json.dumps(caps.to_dict()))["native_bf16"] is True


class TestPurity:
    def test_the_policy_layer_does_not_import_torch_at_module_scope(self):
        """The pure/shell split is what keeps these tests GPU-free. A
        module-level ``import torch`` would bind the conftest MagicMock at
        import time and make every assertion here meaningless."""
        import pathlib

        import spectramr.core.device_capabilities as mod

        source = pathlib.Path(mod.__file__).read_text()
        module_level = [
            line for line in source.splitlines() if line.startswith(("import torch", "from torch"))
        ]
        assert module_level == []


class TestProbeUsesTheRightTorchCall:
    """The planted violation for the emulation trap.

    ``is_bf16_supported()`` must never be consulted: its default branch
    allocates a tensor and reports emulation as support, which is exactly the
    wrong answer on the hardware this project targets.
    """

    def test_the_module_never_calls_is_bf16_supported(self):
        import pathlib

        import spectramr.core.device_capabilities as mod

        source = pathlib.Path(mod.__file__).read_text()
        code = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith(("#", "*"))
        )
        # The docstring explains the trap, so only the CALL form is forbidden.
        assert "is_bf16_supported(" not in code.split('"""')[-1]

    def test_the_probe_reads_get_device_capability(self):
        import pathlib

        import spectramr.core.device_capabilities as mod

        source = pathlib.Path(mod.__file__).read_text()
        assert "get_device_capability" in source


class TestProbeFallsBackToTheDeclaredTarget:
    def test_a_declared_target_is_used_when_nothing_is_visible(self):
        """The audit runs on a login node; the run lands somewhere else."""
        from spectramr.core.device_capabilities import probe_device_capabilities

        caps = probe_device_capabilities("cpu", environ={TARGET_CAPABILITY_ENV: "7.0"})
        assert caps.capability == (7, 0)
        assert caps.source == f"env:{TARGET_CAPABILITY_ENV}"
        assert caps.native_bf16 is False

    def test_a_malformed_target_is_reported_not_obeyed(self):
        """Never raises: this is the reporting path."""
        from spectramr.core.device_capabilities import probe_device_capabilities

        caps = probe_device_capabilities("cpu", environ={TARGET_CAPABILITY_ENV: "sm_70"})
        assert caps.capability is None

    def test_the_record_type_is_returned(self):
        from spectramr.core.device_capabilities import probe_device_capabilities

        assert isinstance(probe_device_capabilities("cpu", environ={}), DeviceCapabilities)
