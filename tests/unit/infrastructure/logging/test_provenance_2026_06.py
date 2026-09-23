"""Tests for run-provenance & traceability capture (2026-06-11).

``infrastructure/logging/provenance.py`` is the SSOT for "what code / what
machine / when / how big" — the metadata that ties a cluster result back to
the commit that produced it. Every helper must be fail-open: a missing git
binary, a broken torch, or an un-dumpable config degrades one field, never
the run.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from spectramr.infrastructure.logging import provenance as prov


# --------------------------------------------------------------------------- #
# torch / env probe
# --------------------------------------------------------------------------- #
def test_torch_runtime_has_available_key():
    info = prov.torch_runtime()
    assert "available" in info
    # When torch is present the probe must report the version + cuda flags.
    if info["available"]:
        assert "version" in info
        assert "cuda_available" in info
        assert isinstance(info["devices"], list)


def test_runtime_environment_shape():
    env = prov.runtime_environment()
    for key in ("python", "platform", "hostname", "pid", "user", "torch"):
        assert key in env, key
    assert isinstance(env["pid"], int)
    assert isinstance(env["torch"], dict)


# --------------------------------------------------------------------------- #
# git provenance
# --------------------------------------------------------------------------- #
def test_git_provenance_available_in_repo():
    g = prov.git_provenance()
    # This test runs inside the repo, so git must resolve.
    assert g["available"] is True
    assert len(g["sha"]) == 40
    assert g["sha_short"] == g["sha"][:12]
    assert isinstance(g["dirty"], bool)


def test_git_provenance_fail_open_outside_repo(tmp_path):
    # A non-repo directory must degrade to available=False, never raise.
    g = prov.git_provenance(repo_dir=str(tmp_path))
    assert g == {"available": False}


# --------------------------------------------------------------------------- #
# parameter counting
# --------------------------------------------------------------------------- #
def test_count_parameters_totals_and_frozen():
    import torch.nn as nn

    model = nn.Sequential(nn.Conv2d(2, 8, 3), nn.Conv2d(8, 2, 3))
    # Freeze the first conv's params.
    for p in model[0].parameters():
        p.requires_grad_(False)

    counts = prov.count_parameters(model)
    assert counts["total"] == counts["trainable"] + counts["frozen"]
    assert counts["frozen"] > 0
    assert counts["size_mb"] >= 0.0


def test_count_parameters_fail_open_on_none():
    assert prov.count_parameters(None) == {}


# --------------------------------------------------------------------------- #
# config fingerprint + effective batch
# --------------------------------------------------------------------------- #
def test_config_fingerprint_is_stable_and_sensitive():
    class _Cfg:
        def __init__(self, lr):
            self._lr = lr

        def model_dump(self, mode="json"):
            return {"optimization": {"learning_rate": self._lr}}

    a = prov.config_fingerprint(_Cfg(1e-4))
    b = prov.config_fingerprint(_Cfg(1e-4))
    c = prov.config_fingerprint(_Cfg(2e-4))
    assert a == b, "same resolved config ⇒ same fingerprint"
    assert a != c, "different knobs ⇒ different fingerprint"
    assert len(a) == 12


def test_config_fingerprint_fail_open():
    assert prov.config_fingerprint(object()) is None


def test_effective_batch_size_multiplies(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    config = SimpleNamespace(
        data=SimpleNamespace(loader=SimpleNamespace(batch_size=8)),
        optimization=SimpleNamespace(gradient=SimpleNamespace(accumulation_steps=2)),
    )
    eb = prov.effective_batch_size(config)
    assert eb["per_device"] == 8
    assert eb["grad_accum"] == 2
    assert eb["world_size"] == 4
    assert eb["effective"] == 64


def test_effective_batch_size_defaults_world_one(monkeypatch):
    for k in ("WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS"):
        monkeypatch.delenv(k, raising=False)
    config = SimpleNamespace(
        data=SimpleNamespace(loader=SimpleNamespace(batch_size=4)),
        optimization=SimpleNamespace(gradient=SimpleNamespace(accumulation_steps=1)),
    )
    assert prov.effective_batch_size(config)["effective"] == 4


# --------------------------------------------------------------------------- #
# run id + composite record
# --------------------------------------------------------------------------- #
def test_make_run_id_slugifies_and_stamps():
    rid = prov.make_run_id("exp 11/cold diff", datetime(2026, 6, 11, 20, 5, 0), "abc123")
    assert rid == "exp-11-cold-diff-20260611_200500-abc123"


def test_make_run_id_handles_missing_git():
    rid = prov.make_run_id("x", datetime(2026, 6, 11, 20, 5, 0), None)
    assert rid.endswith("-nogit")


def test_collect_run_provenance_composite():
    config = SimpleNamespace(
        data=SimpleNamespace(loader=SimpleNamespace(batch_size=2)),
        optimization=SimpleNamespace(gradient=SimpleNamespace(accumulation_steps=1)),
        model=SimpleNamespace(model_type="unet"),
        training=SimpleNamespace(training_mode="reconstruction"),
        model_dump=lambda mode="json": {"a": 1},
    )
    rec = prov.collect_run_provenance(
        config, seed=7, device="cuda", run_name="myrun"
    )
    assert rec["run_name"] == "myrun"
    assert rec["seed"] == 7
    assert rec["device"] == "cuda"
    assert rec["model_type"] == "unet"
    assert rec["training_mode"] == "reconstruction"
    assert rec["config_sha256"] is not None
    assert rec["batch"]["per_device"] == 2
    assert "git" in rec and "env" in rec
    assert rec["run_id"].startswith("myrun-")
    assert "launch" not in rec  # not started via `spectramr launch` → no key


def test_collect_run_provenance_folds_in_plugins(monkeypatch):
    """Out-of-tree plugin sources (SPECTRAMR_PLUGINS + config.plugins.paths) are
    stamped into the record (pitfall #15c — the resolved knob is traceable).
    This was shipped untested; M3 closes the gap."""
    monkeypatch.setenv("SPECTRAMR_PLUGINS", "mypkg.models.foo")
    config = SimpleNamespace(
        data=SimpleNamespace(loader=SimpleNamespace(batch_size=2)),
        optimization=SimpleNamespace(gradient=SimpleNamespace(accumulation_steps=1)),
        model=SimpleNamespace(model_type="unet"),
        training=SimpleNamespace(training_mode="reconstruction"),
        plugins=SimpleNamespace(paths=["mycfg.plugin"]),
        model_dump=lambda mode="json": {"a": 1},
    )
    rec = prov.collect_run_provenance(config, seed=1, device="cpu", run_name="r")
    assert "plugins" in rec, "plugin provenance must be stamped (pitfall #15c)"
    assert "mypkg.models.foo" in rec["plugins"]["env_var"]
    assert rec["plugins"]["config_paths"] == ["mycfg.plugin"]


def test_collect_run_provenance_folds_in_launch(monkeypatch):
    """When started via ``spectramr launch`` (SPECTRAMR_LAUNCH_* present), the
    resolved backend + resources are folded into the record (pitfall #15c)."""
    monkeypatch.setenv("SPECTRAMR_LAUNCH_BACKEND", "docker")
    monkeypatch.setenv("SPECTRAMR_LAUNCH_GPUS", "2")
    monkeypatch.setenv("SPECTRAMR_LAUNCH_ACCOUNT", "acct")
    config = SimpleNamespace(
        data=SimpleNamespace(loader=SimpleNamespace(batch_size=2)),
        optimization=SimpleNamespace(gradient=SimpleNamespace(accumulation_steps=1)),
        model=SimpleNamespace(model_type="unet"),
        training=SimpleNamespace(training_mode="reconstruction"),
        model_dump=lambda mode="json": {"a": 1},
    )
    rec = prov.collect_run_provenance(config, seed=7, device="cpu", run_name="r")
    assert rec["launch"] == {"backend": "docker", "gpus": 2, "account": "acct"}


# --------------------------------------------------------------------------- #
# banner rendering
# --------------------------------------------------------------------------- #
def test_format_provenance_lines_renders_all_sections():
    rec = {
        "run_id": "myrun-20260611_200500-abc",
        "model_type": "unet",
        "config_sha256": "deadbeef0123",
        "seed": 7,
        "device": "cuda",
        "git": {"available": True, "sha_short": "abc", "branch": "dev", "dirty": True},
        "env": {
            "hostname": "node07",
            "pid": 4242,
            "user": "researcher",
            "python": "3.12.12",
            "torch": {
                "available": True,
                "version": "2.11.0",
                "devices": [{"name": "A100", "total_mem_gb": 40.0}],
            },
        },
        "model": {"total": 12_000_000, "size_mb": 48.0, "frozen": 2_000_000},
        "data": {"train": 100, "val": 20},
        "batch": {"per_device": 2, "grad_accum": 1, "world_size": 2, "effective": 4},
        "slurm": {"SLURM_JOB_ID": "999", "SLURM_NODELIST": "node07"},
    }
    lines = prov.format_provenance_lines(rec)
    blob = "\n".join(lines)
    assert "myrun-20260611_200500-abc" in blob
    assert "abc @ dev (DIRTY)" in blob
    assert "node07" in blob
    assert "A100" in blob
    assert "12.00M" in blob  # model params in millions
    # The data block gained explicit units (_format_split_counts): a bare
    # int is the pre-units shape and still renders, as "batches".
    assert "train[batches=100]" in blob
    assert "= 4" in blob  # effective batch
    assert "job=999" in blob


def test_format_provenance_lines_renders_unit_labelled_split_counts():
    """The shape ``describe_dataloader`` produces today: unit -> number.

    The sibling test above feeds the pre-units bare int, so without this the
    modern shape had no assertion at all.
    """
    blob = "\n".join(
        prov.format_provenance_lines(
            {
                "git": {"available": False},
                "data": {"train": {"samples": 100, "batches": 25}},
            }
        )
    )
    assert "train[samples=100, batches=25]" in blob


def test_format_provenance_lines_handles_missing_fields():
    # An empty-ish record must not raise (fail-open rendering).
    lines = prov.format_provenance_lines({"git": {"available": False}})
    blob = "\n".join(lines)
    assert "git        : unavailable" in blob


def test_log_provenance_emits_and_never_raises():
    import logging

    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("test.prov")
    log.setLevel(logging.INFO)
    log.addHandler(_H())
    prov.log_provenance({"git": {"available": False}}, logger=log)
    assert any("run provenance" in m for m in records)


class TestTheLossObjectiveIsStamped:
    """Non-negotiable 8's third obligation, for the one thing a run optimises.

    ``LossWeightTable.provenance()`` called itself "the stamp written into the
    run record" and had no caller outside a unit test -- the advertised-but-
    unwired shape, in the mechanism meant to enforce wiring. The YAML is not a
    substitute: a weight is declarable on two surfaces, ``lambda_mse`` files
    under ``l2``, an equal-weight duplicate reconciles into one entry naming
    both paths, and a warm-up-gated term resolves to 0.0 for its first N steps.
    """

    ARM = (
        "experiments/inprogress/kspace_filling/attention_shootout/"
        "experiment_11_attention_none.yaml"
    )

    def _record(self):
        import pathlib

        import pytest as _pytest

        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.logging.provenance import collect_run_provenance

        if not pathlib.Path(self.ARM).exists():
            _pytest.skip("kspace_filling cohort not present")
        cfg = TrainingSettings.from_yaml(self.ARM)
        return collect_run_provenance(config=cfg, seed=42, device="cpu", run_name="t")

    def test_the_block_reaches_the_run_record(self):
        assert "losses" in self._record()

    def test_it_carries_the_semantics_version(self):
        """Without it, two records are not comparable across a change in what a
        weight MEANS -- which is the failure a stamp exists to prevent."""
        assert self._record()["losses"]["semantics_version"] is not None

    def test_every_resolved_term_names_the_declaration_it_came_from(self):
        """The `source` is the half the YAML cannot supply: it says which of the
        two surfaces won, after canonicalisation."""
        resolved = self._record()["losses"]["resolved"]
        assert resolved, "no terms stamped"
        for name, spec in resolved.items():
            assert spec["source"], f"{name} stamped with no declaration source"
            assert set(spec) == {"weight", "enabled", "source", "warmup_gated"}

    def test_a_strategy_inline_lambda_is_stamped_beside_a_list_entry(self):
        """The two declaration surfaces reach the same record."""
        resolved = self._record()["losses"]["resolved"]
        assert resolved["pre_dc_kspace"]["source"].endswith("lambda_pre_dc_kspace")
        assert "kspace_losses[" in resolved["complex_l1"]["source"]

    def test_provenance_never_blocks_the_run(self, monkeypatch):
        """A capture failure is logged, not raised -- the record is a by-product
        of training, never a precondition for it."""
        import spectramr.models.losses.weights as _weights
        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.logging import provenance as _prov

        def boom(*_a, **_k):
            raise RuntimeError("planted")

        monkeypatch.setattr(_weights, "build_loss_weight_table", boom)
        cfg = TrainingSettings.from_yaml(self.ARM)
        record = _prov.collect_run_provenance(config=cfg, seed=1, device="cpu", run_name="t")
        assert "losses" not in record
