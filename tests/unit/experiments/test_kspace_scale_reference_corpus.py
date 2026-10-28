"""Which ``experiments/inprogress`` arms measure the k-space scale on the input.

``data.processing.kspace_scale_reference`` defaults to ``"target"``, the order
every arm trained under before the knob existed, so an arm that does not
declare it is unchanged. The n2n cohort and every M4Raw ``kspace_filling`` arm
that normalizes k-space declare ``"input"``, because prediction measures the
scale on the input (deployment has no target; #2754, #2768). The fastMRI ports
drop it: their target is the acquisition itself. These tests read YAML and
resolve the schema; none builds a model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from spectramr.config.schemas.data import DataConfigSchema
from spectramr.config.settings import TrainingSettings
from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

_INPROGRESS = Path(__file__).resolve().parents[3] / "experiments" / "inprogress"
_N2N = _INPROGRESS / "n2n"
_N2N_ARMS = sorted(_N2N.glob("n2n_*.yaml"))
_KSPACE_FILLING = _INPROGRESS / "kspace_filling"
#: An arm outside both cohorts, loaded end to end to show the default reaches it.
_CONTROL = (
    _INPROGRESS / "kspace_filling_fastmri_brain" / "fastmri_brain_experiment_11_kfn_none.yaml"
)


def _declared_reference(cfg: Any) -> Any:
    if not isinstance(cfg, dict):
        return None
    processing = (cfg.get("data") or {}).get("processing") or {}
    return processing.get("kspace_scale_reference") if isinstance(processing, dict) else None


def _declarers(arms: dict[Path, Any]) -> set[Path]:
    """The arms that declare the knob at all."""
    return {path for path, cfg in arms.items() if _declared_reference(cfg) is not None}


def _normalizes_kspace(cfg: Any) -> bool:
    processing = (cfg.get("data") or {}).get("processing") or {} if isinstance(cfg, dict) else {}
    return bool(isinstance(processing, dict) and processing.get("enable_kspace_normalization"))


def _expected_declarers(arms: dict[Path, Any]) -> set[Path]:
    """The n2n arms, and the M4Raw kspace_filling arms that normalize k-space."""
    filling = {
        path
        for path, cfg in arms.items()
        if _KSPACE_FILLING in path.parents and _normalizes_kspace(cfg)
    }
    return set(_N2N_ARMS) | filling


def _corpus() -> dict[Path, Any]:
    out: dict[Path, Any] = {}
    for path in sorted(_INPROGRESS.rglob("*.yaml")):
        try:
            out[path] = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue  # not this test's concern; the load gate owns unparsable files
    return out


def test_the_schema_default_is_the_historical_target_order() -> None:
    assert DataConfigSchema().processing.kspace_scale_reference == "target"


@pytest.mark.skipif(not _INPROGRESS.is_dir(), reason="corpus not present")
def test_exactly_the_input_scaled_cohorts_declare_the_reference() -> None:
    """n2n and the normalizing kspace_filling arms declare it; every other arm omits it."""
    corpus = _corpus()
    declarers, expected = _declarers(corpus), _expected_declarers(corpus)
    assert declarers == expected, sorted(str(p) for p in declarers ^ expected)
    assert all(_declared_reference(corpus[p]) == "input" for p in declarers)


def test_a_normalizing_kspace_filling_arm_without_the_reference_is_caught() -> None:
    """PLANTED: a kspace_filling arm that normalizes but omits the knob."""
    arm = _KSPACE_FILLING / "planted.yaml"
    planted = {arm: {"data": {"processing": {"enable_kspace_normalization": True}}}}
    assert (_expected_declarers(planted) - set(_N2N_ARMS)) - _declarers(planted) == {arm}


def test_the_declarer_predicate_sees_a_planted_arm() -> None:
    """PLANTED: an arm outside the cohort that declares the knob is caught."""
    planted = {
        Path("self_supervised/x.yaml"): {
            "data": {"processing": {"kspace_scale_reference": "input"}}
        },
        Path("other/y.yaml"): {"data": {"processing": {"kspace_scale_domain": "image"}}},
        Path("other/z.yaml"): None,
    }
    assert _declarers(planted) == {Path("self_supervised/x.yaml")}


@pytest.mark.skipif(not _CONTROL.is_file(), reason="control arm not present")
def test_an_undeclaring_arm_resolves_to_target() -> None:
    config = TrainingSettings.from_yaml(_CONTROL)
    assert config.data.processing.kspace_scale_reference == "target"
    assert KSpaceNormalizationSpec.from_data_config(config.data).scale_reference == "target"


@pytest.mark.skipif(not _N2N_ARMS, reason="n2n cohort not present")
def test_the_cohort_has_twelve_arms() -> None:
    """The eleven, plus arm D's barrier at weight 0.1."""
    assert len(_N2N_ARMS) == 12, [p.stem for p in _N2N_ARMS]


@pytest.mark.parametrize("arm", _N2N_ARMS, ids=lambda p: p.stem)
def test_every_n2n_arm_loads_and_resolves_input(arm: Path) -> None:
    config = TrainingSettings.from_yaml(arm)
    assert config.data.processing.kspace_scale_reference == "input"
    assert KSpaceNormalizationSpec.from_data_config(config.data).scale_reference == "input"


class _ConfigProxy:
    """The shape ``DataPipelineDirector`` hands ``from_training_config``.

    The production proxy is local to the director's method, so this repeats
    its two lines: ``undersampling`` set explicitly, everything else delegated.
    """

    def __init__(self, data: Any, undersampling: Any) -> None:
        self._data = data
        self.undersampling = undersampling

    def __getattr__(self, name: str) -> Any:
        return getattr(self._data, name)


def _built_references(config: TrainingSettings) -> dict[str, list[str]]:
    from spectramr.data.builders.torchio_transform_builder import (
        TorchIOTransformBuilder,
        TorchIOTransformConfig,
    )
    from spectramr.data.transforms.normalization import KSpaceNormalizationTransform

    tcfg = TorchIOTransformConfig.from_training_config(
        _ConfigProxy(config.data, config.undersampling)
    )
    chains = {
        "train": TorchIOTransformBuilder.build_train_transforms(tcfg),
        "val": TorchIOTransformBuilder.build_val_transforms(tcfg),
    }
    return {
        split: [
            t.spec.scale_reference
            for t in chain.transforms
            if isinstance(t, KSpaceNormalizationTransform)
        ]
        for split, chain in chains.items()
    }


@pytest.mark.parametrize("arm", _N2N_ARMS, ids=lambda p: p.stem)
def test_the_built_n2n_chains_measure_on_the_input(arm: Path) -> None:
    """The transform that runs, not just the resolved config, carries "input"."""
    assert _built_references(TrainingSettings.from_yaml(arm)) == {
        "train": ["input"],
        "val": ["input"],
    }


@pytest.mark.skipif(not _N2N_ARMS, reason="n2n cohort not present")
def test_the_built_chain_follows_an_undeclared_reference(tmp_path: Path) -> None:
    """PLANTED: the same arm without the key builds a target-first chain."""
    raw = yaml.safe_load(_N2N_ARMS[0].read_text())
    del raw["data"]["processing"]["kspace_scale_reference"]
    planted = tmp_path / _N2N_ARMS[0].name
    planted.write_text(yaml.safe_dump(raw))
    config = TrainingSettings.from_yaml(planted)
    assert _built_references(config) == {"train": ["target"], "val": ["target"]}
