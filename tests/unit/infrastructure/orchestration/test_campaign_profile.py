"""``campaign profile``: which arms are profiled, where each profile is filed, and what is refused."""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.infrastructure.orchestration import campaign_profile as cp


def _campaign(tmp_path: Path, execution: str = "array", missing: tuple[str, ...] = ()) -> Path:
    lines = [f"name: camp\nexecution: {execution}\noutput_dir: out/camp\nexperiments:"]
    for name, stem in (("Arm A", "arm_a"), ("Arm B", "arm_b")):
        if stem not in missing:
            (tmp_path / f"{stem}.yaml").write_text("x: 1\n")
        lines.append(f"- name: {name}\n  config: {stem}.yaml")
    path = tmp_path / "camp.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize(
    ("execution", "dirs"), [("array", ["arm_a", "arm_b"]), ("parallel", ["Arm A", "Arm B"])]
)
def test_each_arm_is_filed_where_its_training_run_writes(tmp_path, execution, dirs):
    _, campaign_dir, plans = cp.plan_profile(_campaign(tmp_path, execution), tmp_path)
    assert campaign_dir == tmp_path.resolve() / "out/camp"
    assert [p.run_dir for p in plans] == [str(campaign_dir / d) for d in dirs]
    assert all(p.skip_reason is None for p in plans)


def test_a_missing_config_is_reported_not_dropped(tmp_path):
    _, _, plans = cp.plan_profile(_campaign(tmp_path, missing=("arm_b",)), tmp_path)
    assert [p.skip_reason is None for p in plans] == [True, False]
    assert "config not found" in plans[1].skip_reason


def test_only_selects_by_name_or_run_dir_and_rejects_unknown_names(tmp_path):
    camp = _campaign(tmp_path)
    assert [p.name for p in cp.plan_profile(camp, tmp_path, only=["arm_b"])[2]] == ["Arm B"]
    assert [p.name for p in cp.plan_profile(camp, tmp_path, only=["Arm A"])[2]] == ["Arm A"]
    with pytest.raises(ValueError, match="names no arm"):
        cp.plan_profile(camp, tmp_path, only=["nope"])


def test_a_dry_run_writes_nothing_and_renders_a_profile_array(tmp_path, monkeypatch):
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "acct")
    cfg, campaign_dir, plans = cp.plan_profile(_campaign(tmp_path), tmp_path)
    job, script = cp.submit_profile(
        cfg, campaign_dir, plans, tmp_path, profile_args=["--mode", "cpu-only"], dry_run=True
    )
    assert job is None and not campaign_dir.exists()
    assert "--verb profile --profile-run-dirs" in script
    call = script[script.index("python -m spectramr.cli.manifest_dispatch") :]
    assert "--output-base" not in call and "--prod" not in call and "RESUME_FLAG" not in script
    assert "#SBATCH --time=00:45:00" in script and "#SBATCH --array=0-1%2" in script
    assert f"{campaign_dir / 'arm_a'}\n{campaign_dir / 'arm_b'}" in script
    assert "--mode\ncpu-only" in script


def test_nothing_runnable_is_refused(tmp_path):
    cfg, campaign_dir, plans = cp.plan_profile(
        _campaign(tmp_path, missing=("arm_a", "arm_b")), tmp_path
    )
    with pytest.raises(ValueError, match="nothing to submit"):
        cp.submit_profile(cfg, campaign_dir, plans, tmp_path, profile_args=[], dry_run=True)
