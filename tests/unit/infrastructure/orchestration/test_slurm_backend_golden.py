"""Golden-file guard for the campaign SLURM job-script generator (WS-E).

The campaign orchestrator's ``generate_job_script`` emits sbatch for *live*
cluster runs. WS-E consolidates its ``#SBATCH`` header onto the shared
:class:`spectramr.infrastructure.execution.SlurmBackend` so the directive format
stops drifting from the launcher's. This test pins the generated script
byte-for-byte against a snapshot captured BEFORE the refactor, so the
consolidation cannot change what lands on the cluster.

If a *deliberate* change to the script is made, regenerate the fixture and
review the diff.
"""

from __future__ import annotations

from pathlib import Path

from spectramr.infrastructure.orchestration.slurm_backend import SLURMBackend

_GOLDEN = Path(__file__).parent / "fixtures" / "golden_campaign_sbatch.txt"


def test_generate_job_script_byte_identical_to_golden(monkeypatch):
    # Deterministic: the shared ResourceSpec consults these env vars. The
    # account has no default and is now required, so pin it to a placeholder
    # rather than unsetting it; partition and mail-user stay unset so the
    # snapshot shows the no-optional-directives shape.
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")
    monkeypatch.delenv("SPECTRAMR_SLURM_PARTITION", raising=False)
    monkeypatch.delenv("SPECTRAMR_SLURM_MAIL_USER", raising=False)

    out = SLURMBackend.generate_job_script(
        experiment_name="exp_demo",
        config_path="experiments/inprogress/x.yaml",
        output_dir="experiments/results/campaigns/demo/exp_demo",
        base_dir="/scratch/myalloc/spectramr",
        slurm_params={"gpus": 2, "mem": "64GB", "time": "96:00:00"},
        resume=True,
        test_manifest="data/manifests/test.txt",
        config_overrides={"checkpoint.inject_as": "model.pretrained"},
    )
    assert out == _GOLDEN.read_text(), (
        "campaign sbatch script changed vs golden snapshot — if intentional, "
        "regenerate tests/unit/infrastructure/orchestration/fixtures/"
        "golden_campaign_sbatch.txt and review the diff."
    )


def test_generate_job_script_emits_partition_when_env_set(monkeypatch):
    """M2: pin the WS-E env-conditional partition behavior as CONSCIOUS, not drift.

    The byte-for-byte golden above is captured with ``SPECTRAMR_SLURM_PARTITION``
    UNSET, so it only proves "the consolidation changes nothing" on a
    partition-less cluster (the M4Raw allocation, where the env is unset). When
    the env IS set, the consolidated generator's shared ``ResourceSpec``
    env-defaults the partition and emits a ``#SBATCH --partition=`` line — which
    the pre-refactor inline path (hardcoded ``partition=None``) never did. This
    test pins that as an intentional, tested behavior so it can't silently drift
    onto a cluster.
    """
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")
    monkeypatch.setenv("SPECTRAMR_SLURM_PARTITION", "gpu-a100")
    out = SLURMBackend.generate_job_script(
        experiment_name="exp_demo",
        config_path="experiments/inprogress/x.yaml",
        output_dir="experiments/results/campaigns/demo/exp_demo",
        base_dir="/scratch/myalloc/spectramr",
        slurm_params={"gpus": 2, "mem": "64GB", "time": "96:00:00"},
        resume=True,
        test_manifest="data/manifests/test.txt",
        config_overrides={"checkpoint.inject_as": "model.pretrained"},
    )
    assert "#SBATCH --partition=gpu-a100" in out
    # ...and the env-unset golden snapshot must carry NO partition line.
    assert "--partition" not in _GOLDEN.read_text()


def test_golden_carries_no_site_identity(monkeypatch):
    """The snapshot ships in the public tree, so it must name no real site.

    The pre-#1146 golden hardcoded an allocation account and a university mail
    domain; both were inputs nobody outside one cluster could use. This asserts
    the placeholder shape survives a regeneration -- a re-baked default would
    otherwise re-enter the tree through the fixture rather than the source.
    """
    text = _GOLDEN.read_text()
    assert "--mail-user" not in text
    assert "#SBATCH --account=test_alloc" in text
    for leaked in ("johnsson", "agdihi", "cougarnet", "uh.edu"):
        assert leaked not in text, f"golden fixture leaks {leaked!r}"


def test_mail_user_directive_is_emitted_when_configured(monkeypatch):
    """Discrimination leg: the golden's missing line is 'unset', not 'dropped'."""
    monkeypatch.setenv("SPECTRAMR_SLURM_ACCOUNT", "test_alloc")
    monkeypatch.delenv("SPECTRAMR_SLURM_PARTITION", raising=False)
    monkeypatch.setenv("SPECTRAMR_SLURM_MAIL_USER", "me@example.org")
    out = SLURMBackend.generate_job_script(
        experiment_name="exp_demo",
        config_path="experiments/inprogress/x.yaml",
        output_dir="experiments/results/campaigns/demo/exp_demo",
        base_dir="/scratch/myalloc/spectramr",
        slurm_params={"gpus": 2, "mem": "64GB", "time": "96:00:00"},
        resume=True,
        test_manifest="data/manifests/test.txt",
        config_overrides={"checkpoint.inject_as": "model.pretrained"},
    )
    assert "#SBATCH --mail-user=me@example.org" in out
