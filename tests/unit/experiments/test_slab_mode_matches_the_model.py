"""No in-progress arm may fold a validation volume into channels its model does not take.

`data.sampling.enable_slab_mode` collates a 5D volume to [B, C*D, H, W]. Training samples
depth-1 patches, so for them it changes nothing; validation loads whole volumes, and an
M4Raw volume of 18 slices becomes 18 channels. Experiment 32a declared the flag with a
one-channel model and failed all 45 validation batches on the first sabine run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ARMS = sorted(Path(__file__).resolve().parents[3].glob("experiments/inprogress/**/*.yaml"))


def _slab_arms_with_depth_one_patches() -> list[Path]:
    flagged = []
    for path in ARMS:
        try:
            cfg = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        data = cfg.get("data") if isinstance(cfg, dict) else None
        if not isinstance(data, dict):
            continue
        sampling = data.get("sampling") or {}
        patch = sampling.get("patch_size") or []
        if sampling.get("enable_slab_mode") is True and list(patch)[-1:] == [1]:
            flagged.append(path)
    return flagged


def test_the_corpus_is_found() -> None:
    assert len(ARMS) > 100


def test_no_arm_declares_slab_collation_over_depth_one_patches() -> None:
    """Over depth-1 patches the flag is a no-op in training and an 18-channel fold in validation."""
    names = [str(p.relative_to(p.parents[3])) for p in _slab_arms_with_depth_one_patches()]
    assert not names, f"enable_slab_mode: true with a depth-1 patch: {names}"


@pytest.mark.parametrize(
    "arm",
    [
        "vae_latent/experiment_32a_vae_hf_multicontrast.yaml",
        "physics_driven/experiment_50_strict_physical_disentangled.yaml",
        "physics_driven/experiment_51_fno_epg_multicontrast.yaml",
    ],
)
def test_the_m4raw_disentangled_arms_validate_slice_by_slice(arm: str) -> None:
    root = Path(__file__).resolve().parents[3] / "experiments/inprogress"
    cfg = yaml.safe_load((root / arm).read_text())
    assert cfg["data"]["sampling"]["enable_slab_mode"] is False
