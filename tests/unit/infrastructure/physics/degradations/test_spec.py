"""DegradationSpec contract: every declaration required, keys consistent, errors well-formed."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from spectramr.config.schemas.enums import ArtifactOrigin
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.spec import (
    ActivationGate,
    DegradationSpec,
    MissingInput,
    Realisation,
    Requirement,
    SliceKernel,
    VoxelSliceKernel,
    UnsupportedAcquisitionMode,
    entry,
)
from spectramr.infrastructure.physics.severity import PhysicalParam, SeveritySpec

M = AcquisitionMode
R = Realisation


def _identity(x: torch.Tensor, theta: float, seed: int, geometry: Any, **_: Any) -> torch.Tensor:
    return x


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "code": "D1",
        "name": "toy",
        "family": "noise",
        "description": "a toy",
        "severity": SeveritySpec((PhysicalParam("sigma", "a.u.", 0.0, 1.0),)),
        "affects": frozenset({"magnitude", "phase"}),
        "origin": ArtifactOrigin.SCANNER,
        "implementations": {M.SLICE_2D: _identity, M.MULTISLICE_2D: _identity},
        "realisation": {M.SLICE_2D: R.NATIVE, M.MULTISLICE_2D: R.PER_SLICE_EXACT},
        "requires": frozenset(),
        "condition": None,
        "active_when": frozenset(),
    }
    base.update(overrides)
    return base


def test_enum_members() -> None:
    assert {r.value for r in Realisation} == {"native", "per_slice_exact", "per_slice_independent"}
    assert {r.value for r in Requirement} == {
        "multicoil",
        "qmaps",
        "fat_map",
        "vessel_map",
        "trajectory_radial",
        "sequence_epi",
    }
    assert {g.value for g in ActivationGate} == {"sequence_epi", "accelerated", "fat_present"}


def test_entry_builds_spec_with_properties() -> None:
    spec = entry(**_kwargs())
    assert isinstance(spec, DegradationSpec)
    assert spec.supported_modes == frozenset({M.SLICE_2D, M.MULTISLICE_2D})
    assert not spec.phase_only
    assert entry(**_kwargs(affects=frozenset({"phase"}))).phase_only


@pytest.mark.parametrize("missing", list(_kwargs()))
def test_entry_missing_any_declaration_fails(missing: str) -> None:
    kwargs = _kwargs()
    del kwargs[missing]
    with pytest.raises(TypeError):
        entry(**kwargs)


def test_entry_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        entry(*_kwargs().values())  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"realisation": {M.SLICE_2D: R.NATIVE}},
        {"implementations": {}, "realisation": {}},
        {"affects": frozenset()},
        {"affects": frozenset({"magnitude", "texture"})},
        {
            "implementations": {M.MULTISLICE_2D: _identity},
            "realisation": {M.MULTISLICE_2D: R.PER_SLICE_EXACT},
        },
        {"realisation": {M.SLICE_2D: R.PER_SLICE_EXACT, M.MULTISLICE_2D: R.NATIVE}},
        {"requires": frozenset({"multicoil"})},
        {"active_when": frozenset({"sequence_epi"})},
        {"condition": "not callable"},
    ],
)
def test_planted_bad_spec_raises(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        entry(**_kwargs(**overrides))


def test_spec_may_omit_slice_2d_when_all_native() -> None:
    spec = entry(
        **_kwargs(
            implementations={M.MULTISLICE_2D: _identity},
            realisation={M.MULTISLICE_2D: R.NATIVE},
        )
    )
    assert spec.supported_modes == frozenset({M.MULTISLICE_2D})


def test_spec_mappings_are_read_only() -> None:
    spec = entry(**_kwargs())
    with pytest.raises(TypeError):
        spec.implementations[M.VOLUME_3D] = _identity  # type: ignore[index]


def test_errors_are_value_errors_with_context() -> None:
    err = UnsupportedAcquisitionMode("toy", M.VOLUME_3D, frozenset({M.SLICE_2D, M.MULTISLICE_2D}))
    assert isinstance(err, ValueError)
    msg = str(err)
    assert "toy" in msg and "volume_3d" in msg and "slice_2d" in msg and "multislice_2d" in msg
    missing = MissingInput("toy", Requirement.QMAPS)
    assert isinstance(missing, ValueError)
    assert "toy" in str(missing) and "qmaps" in str(missing)


def _kernel(x: torch.Tensor, theta: float, seed: int = 0, gain: float = 1.0) -> torch.Tensor:
    return x * (theta + seed + gain)


def test_slice_kernel_forwards_seed_and_kwargs_and_ignores_geometry() -> None:
    x = torch.ones(1, 1, 2, 2)
    out = SliceKernel(_kernel)(x, 0.5, 2, object(), gain=3.0)
    assert torch.equal(out, x * 5.5)


def _mm_kernel(x, theta, seed=0, voxel_mm=None):
    return x * (1.0 if voxel_mm is None else voxel_mm[1])


def test_voxel_slice_kernel_forwards_the_geometry_voxel_size() -> None:
    x = torch.ones(1, 1, 2, 2)
    geom = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=(3.0, 0.5, 0.5))
    assert torch.equal(VoxelSliceKernel(_mm_kernel)(x, 0.5, 2, geom), x * 0.5)
    none = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=None)
    assert torch.equal(VoxelSliceKernel(_mm_kernel)(x, 0.5, 2, none), x)


def test_voxel_slice_kernel_refuses_a_caller_voxel_size() -> None:
    geom = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=None)
    with pytest.raises(ValueError, match="geometry"):
        VoxelSliceKernel(_mm_kernel)(torch.ones(1, 1, 2, 2), 0.5, 0, geom, voxel_mm=(1, 1, 1))


def test_fn_is_the_bare_slice_kernel_and_affects_flags() -> None:
    spec = entry(
        **_kwargs(
            implementations={M.SLICE_2D: SliceKernel(_kernel)},
            realisation={M.SLICE_2D: R.NATIVE},
            affects=frozenset({"phase"}),
        )
    )
    assert spec.fn is _kernel
    assert spec.affects_phase and not spec.affects_magnitude


def test_fn_raises_without_a_bare_slice_kernel() -> None:
    no_2d = entry(
        **_kwargs(implementations={M.VOLUME_3D: _identity}, realisation={M.VOLUME_3D: R.NATIVE})
    )
    with pytest.raises(UnsupportedAcquisitionMode):
        _ = no_2d.fn
    with pytest.raises(TypeError, match="no bare kernel"):
        _ = entry(**_kwargs()).fn


def test_unsupported_detail_reaches_the_error_and_may_not_name_an_implemented_mode() -> None:
    spec = entry(
        **_kwargs(
            implementations={M.MULTISLICE_2D: _identity},
            realisation={M.MULTISLICE_2D: R.NATIVE},
            unsupported_detail={M.SLICE_2D: "no entry depth"},
        )
    )
    with pytest.raises(UnsupportedAcquisitionMode, match="no entry depth"):
        spec.fn  # noqa: B018
    with pytest.raises(ValueError, match="unsupported_detail"):
        entry(**_kwargs(unsupported_detail={M.SLICE_2D: "wrong"}))


def test_unsupported_detail_defaults_to_empty_and_is_frozen() -> None:
    spec = entry(**_kwargs())
    assert dict(spec.unsupported_detail) == {}
    spec = entry(**_kwargs(unsupported_detail={M.VOLUME_3D: "no slab"}))
    assert spec.unsupported_detail[M.VOLUME_3D] == "no slab"
    with pytest.raises(TypeError):
        spec.unsupported_detail[M.VOLUME_3D] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    ("detail", "match"),
    [
        ({M.MULTISLICE_2D: "reason"}, "implemented mode"),
        ({M.VOLUME_3D: ""}, "reason"),
        ({"volume_3d": "reason"}, "AcquisitionMode"),
    ],
)
def test_unsupported_detail_is_validated(detail: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        entry(**_kwargs(unsupported_detail=detail))


def test_contract_exceptions_round_trip_pickle() -> None:
    # A DataLoader worker re-raises what it unpickles; a lossy reduce turns these into RuntimeError.
    import pickle

    unsupported = UnsupportedAcquisitionMode(
        "slice_pve", AcquisitionMode.VOLUME_3D, frozenset({AcquisitionMode.SLICE_2D}), detail="2-D"
    )
    missing = MissingInput("coil_dropout", Requirement.MULTICOIL)
    for exc in (unsupported, missing):
        back = pickle.loads(pickle.dumps(exc))
        assert type(back) is type(exc) and str(back) == str(exc)
        assert vars(back) == vars(exc)
