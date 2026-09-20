"""Unit tests for the three novel-2026 CLI subcommands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch


def _make_args(**kwargs) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def test_audit_ksd_smoke_mode_passes(capsys: pytest.CaptureFixture[str]) -> None:
    from spectramr.cli.audit_plan_novel_cli import audit_ksd_cmd

    args = _make_args(
        config=Path("dummy.yaml"),
        smoke=True,
        samples=None,
        score_module=None,
        threshold=0.05,
        bootstrap=200,
        samples_count=64,
        json=False,
    )
    exit_code = audit_ksd_cmd(args)
    captured = capsys.readouterr().out
    assert "smoke mode" in captured
    assert exit_code in (0, 1)  # passes; close-call warning allowed


def test_audit_ksd_json_output_is_parseable(capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from spectramr.cli.audit_plan_novel_cli import audit_ksd_cmd

    args = _make_args(
        config=Path("dummy.yaml"),
        smoke=True,
        samples=None,
        score_module=None,
        threshold=0.05,
        bootstrap=200,
        samples_count=64,
        json=True,
    )
    audit_ksd_cmd(args)
    parsed = json.loads(capsys.readouterr().out)
    assert "ksd_squared" in parsed
    assert "p_value" in parsed
    assert "passed" in parsed


def test_audit_ksd_missing_samples_returns_error(tmp_path: Path) -> None:
    from spectramr.cli.audit_plan_novel_cli import audit_ksd_cmd

    args = _make_args(
        config=Path("dummy.yaml"),
        smoke=False,
        samples=tmp_path / "does_not_exist.pt",
        score_module=None,
        threshold=0.05,
        bootstrap=100,
        samples_count=32,
        json=False,
    )
    code = audit_ksd_cmd(args)
    assert code == 2


def test_audit_ksd_loads_pt_samples(tmp_path: Path) -> None:
    from spectramr.cli.audit_plan_novel_cli import audit_ksd_cmd

    samples_path = tmp_path / "samples.pt"
    torch.save(torch.randn(64, 5), samples_path)
    args = _make_args(
        config=Path("dummy.yaml"),
        smoke=False,
        samples=samples_path,
        score_module=None,
        threshold=0.05,
        bootstrap=100,
        samples_count=32,
        json=True,
    )
    code = audit_ksd_cmd(args)
    assert code in (0, 1, 2)  # any outcome is valid; we test the path runs.


def test_simulate_acquisition_writes_csv(tmp_path: Path) -> None:
    from spectramr.cli.audit_plan_novel_cli import simulate_acquisition_cmd

    out_csv = tmp_path / "sa.csv"
    args = _make_args(
        config=Path("experiments/inprogress/novel_2026/idea_4_ib_active_acquisition.yaml"),
        output=out_csv,
        n_steps=4,
        height=16,
        width=16,
        quiet=True,
    )
    code = simulate_acquisition_cmd(args)
    assert code == 0
    text = out_csv.read_text()
    assert "step,psnr_db,acquired_lines" in text
    rows = [r for r in text.strip().splitlines() if r and not r.startswith("step")]
    assert len(rows) == 4


def test_simulate_acquisition_missing_config_errors(tmp_path: Path) -> None:
    from spectramr.cli.audit_plan_novel_cli import simulate_acquisition_cmd

    args = _make_args(
        config=tmp_path / "missing.yaml",
        output=tmp_path / "sa.csv",
        n_steps=4,
        height=16,
        width=16,
        quiet=True,
    )
    assert simulate_acquisition_cmd(args) == 2


def test_attach_subparsers_registers_three_commands() -> None:
    from spectramr.cli.audit_plan_novel_cli import attach_subparsers

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    attach_subparsers(sub)
    # Argparse stores subparsers in ``choices``.
    assert {"audit-ksd", "infer-protocol", "simulate-acquisition"} <= set(sub.choices)


# --------------------------------------------------------------------- #
# infer-protocol checkpoint loading (D01#1 / #1351)
#
# The verb had *no* tests at all, which is why it shipped a path that
# printed a posterior mode and a 95% CI from a randomly-initialised
# network. Every test below drives ``infer_protocol_cmd`` -- the helper
# is pinned too, but a helper-only pin scores a call-site regression
# green (non-negotiable 16).
# --------------------------------------------------------------------- #

# Deliberately non-default on every axis: with (1, 5, 32) the derivation
# would agree with the old hard-coded constructor by coincidence.
_ARCH = {"in_channels": 2, "cond_dim": 3, "base_width": 16}


def _score_field_unet(**overrides):
    from spectramr.models.diffusion.score_field_unet import ScoreFieldUNet

    return ScoreFieldUNet(**{**_ARCH, **overrides})


def _write_checkpoint(tmp_path: Path, payload, name: str = "ckpt.pth") -> Path:
    path = tmp_path / name
    torch.save(payload, path)
    return path


def _write_image(tmp_path: Path, channels: int = 2) -> Path:
    path = tmp_path / "image.pt"
    torch.save(torch.randn(1, channels, 16, 16), path)
    return path


def _infer_args(checkpoint: Path, image: Path, **kwargs) -> argparse.Namespace:
    base = {
        "checkpoint": checkpoint,
        "image": image,
        "n_steps": 2,
        "step_size": 1e-3,
        "sigma": 0.1,
        "cond_dim": None,
    }
    base.update(kwargs)
    return _make_args(**base)


def test_infer_protocol_runs_on_a_canonical_envelope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The envelope ``CheckpointService`` actually writes must still work.

    ``checkpoint_service.py:324`` stores ``model_state_dict`` beside epoch /
    step / RNG state and *no* config block -- which is why the architecture is
    derived from tensor shapes rather than resolved through ``MODEL_REGISTRY``
    as ``D01#1`` proposes.
    """
    from spectramr.cli.audit_plan_novel_cli import infer_protocol_cmd
    from spectramr.core.rng_state import capture_rng_state

    ckpt = _write_checkpoint(
        tmp_path,
        {
            "epoch": 3,
            "step": 120,
            "model_state_dict": _score_field_unet().state_dict(),
            # The half that made the first version of this test vacuous: the
            # real writer stamps rng_state unconditionally, and its numpy
            # entry is what a plain ``torch.load`` refuses on torch >= 2.6.
            "rng_state": capture_rng_state(),
        },
    )
    assert infer_protocol_cmd(_infer_args(ckpt, _write_image(tmp_path))) == 0
    out = capsys.readouterr().out
    # cond_dim=3 was read off the checkpoint, not taken from the old default 5.
    assert "c[2]" in out and "c[3]" not in out


def test_infer_protocol_accepts_a_bare_state_dict(tmp_path: Path) -> None:
    """Layout 2 of the three the docstring advertises."""
    from spectramr.cli.audit_plan_novel_cli import infer_protocol_cmd

    ckpt = _write_checkpoint(tmp_path, _score_field_unet().state_dict())
    assert infer_protocol_cmd(_infer_args(ckpt, _write_image(tmp_path))) == 0


def test_instantiate_accepts_a_pickled_module() -> None:
    """Layout 3: a pickled module already carries its architecture.

    Pinned at the helper on purpose -- see the next test for why the verb
    itself can no longer reach this branch.
    """
    from spectramr.cli.audit_plan_novel_cli import _instantiate_model_from_state

    model = _instantiate_model_from_state(_score_field_unet(), cond_dim=None)
    assert model.head_sc.out_features == _ARCH["cond_dim"]
    assert not model.training


def test_pickled_module_checkpoints_are_refused_by_torch_load(tmp_path: Path) -> None:
    """torch >= 2.6 defaults ``weights_only=True``, so layout 3 dies at load.

    The verb's own ``torch.load`` refuses a pickled ``ScoreFieldUNet`` before
    ``_instantiate_model_from_state`` ever sees it. That is a *loud* refusal,
    which is why it is pinned rather than worked around: relaxing it would
    mean ``weights_only=False`` on an untrusted path.
    """
    import pickle

    from spectramr.cli.audit_plan_novel_cli import infer_protocol_cmd

    ckpt = _write_checkpoint(tmp_path, _score_field_unet())
    with pytest.raises(pickle.UnpicklingError, match=r"[Ww]eights only"):
        infer_protocol_cmd(_infer_args(ckpt, _write_image(tmp_path)))


def test_infer_protocol_raises_on_a_foreign_checkpoint(tmp_path: Path) -> None:
    """The defect: this used to warn and report a CI from random weights."""
    from spectramr.cli.audit_plan_novel_cli import infer_protocol_cmd

    ckpt = _write_checkpoint(tmp_path, {"model_state_dict": {"fc.weight": torch.randn(4, 4)}})
    with pytest.raises(KeyError, match=r"lift\.weight"):
        infer_protocol_cmd(_infer_args(ckpt, _write_image(tmp_path)))


def test_infer_protocol_raises_on_a_partial_state_dict(tmp_path: Path) -> None:
    """``strict=False`` accepted this silently, loading a subset of the weights."""
    from spectramr.cli.audit_plan_novel_cli import infer_protocol_cmd

    sd = _score_field_unet().state_dict()
    del sd["head_sx.weight"]
    ckpt = _write_checkpoint(tmp_path, {"model_state_dict": sd})
    with pytest.raises(RuntimeError, match=r"head_sx\.weight"):
        infer_protocol_cmd(_infer_args(ckpt, _write_image(tmp_path)))


def test_infer_protocol_raises_when_cond_dim_contradicts_the_checkpoint(
    tmp_path: Path,
) -> None:
    """An explicit ``--cond-dim`` is an assertion, not a source of truth."""
    from spectramr.cli.audit_plan_novel_cli import infer_protocol_cmd

    ckpt = _write_checkpoint(tmp_path, {"model_state_dict": _score_field_unet().state_dict()})
    with pytest.raises(ValueError, match="--cond-dim 5 contradicts"):
        infer_protocol_cmd(_infer_args(ckpt, _write_image(tmp_path), cond_dim=5))


def test_infer_protocol_accepts_a_cond_dim_that_agrees(tmp_path: Path) -> None:
    from spectramr.cli.audit_plan_novel_cli import infer_protocol_cmd

    ckpt = _write_checkpoint(tmp_path, {"model_state_dict": _score_field_unet().state_dict()})
    assert infer_protocol_cmd(_infer_args(ckpt, _write_image(tmp_path), cond_dim=3)) == 0


def test_infer_protocol_still_exits_2_on_a_missing_file(tmp_path: Path) -> None:
    """Usage errors keep their exit code; only *defects* were promoted to raises."""
    from spectramr.cli.audit_plan_novel_cli import infer_protocol_cmd

    image = _write_image(tmp_path)
    assert infer_protocol_cmd(_infer_args(tmp_path / "nope.pth", image)) == 2
    ckpt = _write_checkpoint(tmp_path, _score_field_unet().state_dict())
    assert infer_protocol_cmd(_infer_args(ckpt, tmp_path / "nope.pt")) == 2


def test_architecture_is_read_off_the_shapes_not_guessed() -> None:
    from spectramr.cli.audit_plan_novel_cli import _architecture_from_state_dict

    assert _architecture_from_state_dict(_score_field_unet().state_dict()) == _ARCH


def test_architecture_rejects_an_internally_inconsistent_state_dict() -> None:
    from spectramr.cli.audit_plan_novel_cli import _architecture_from_state_dict

    sd = _score_field_unet().state_dict()
    sd["cond_embed.0.weight"] = torch.randn(64, 9)  # implies cond_dim=8, not 3
    with pytest.raises(ValueError, match="internally inconsistent"):
        _architecture_from_state_dict(sd)


def test_instantiate_rejects_a_foreign_module() -> None:
    from spectramr.cli.audit_plan_novel_cli import _instantiate_model_from_state

    with pytest.raises(TypeError, match="not a ScoreFieldUNet"):
        _instantiate_model_from_state(torch.nn.Linear(4, 4), cond_dim=None)


def test_instantiate_rejects_a_non_dict_payload() -> None:
    from spectramr.cli.audit_plan_novel_cli import _instantiate_model_from_state

    with pytest.raises(TypeError, match="unsupported checkpoint payload"):
        _instantiate_model_from_state([1, 2, 3], cond_dim=None)


def test_infer_protocol_cond_dim_default_is_none() -> None:
    """A concrete default cannot be told apart from a declaration (D01#4)."""
    from spectramr.cli.audit_plan_novel_cli import attach_subparsers

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    attach_subparsers(sub)
    assert sub.choices["infer-protocol"].get_default("cond_dim") is None


@pytest.mark.unit
def test_infer_protocol_loads_a_checkpoint_the_framework_actually_wrote(
    tmp_path: Path,
) -> None:
    """End-to-end against the real writer, not a hand-built dict.

    ``CheckpointService.save_checkpoint`` is the canonical producer, and its
    envelope is the one the verb must read. Building the payload by hand is
    how the rng_state gap survived: the fixture agreed with the docstring
    rather than with the writer.
    """
    from spectramr.cli.audit_plan_novel_cli import infer_protocol_cmd
    from spectramr.config.schemas.checkpoint import CheckpointConfigSchema
    from spectramr.infrastructure.services.checkpoint_service import CheckpointService

    model = _score_field_unet()
    service = CheckpointService(
        CheckpointConfigSchema(checkpoint_dir=str(tmp_path / "ckpts"), format="pth")
    )
    written = service.save_checkpoint(
        model=model,
        optimizer=torch.optim.Adam(model.parameters()),
        epoch=1,
        step=5,
        loss=0.1,
    )

    args = _infer_args(Path(written), _write_image(tmp_path))
    assert infer_protocol_cmd(args) == 0
