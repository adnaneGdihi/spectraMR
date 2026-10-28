"""Tests for the base-class generic ``model_output`` debug snapshot
(2026-06-21 forensics "model output missing" gap).

Before this change only the diffusion strategy emitted a ``model_output``
snapshot (it captured its prediction by hand inside ``_compute_losses_impl``);
every other paradigm — the 18 mrixfields field strategies, VF, GAN, VAE,
reconstruction — emitted only the pre-forward ``first_steps`` inputs, so the
forensics contact sheets had no model-output row for them.

The fix gives the base class parity with its existing ``first_steps``
auto-snapshot: a forward hook on ``self.env.generator`` stashes the latest
output, and the ``_compute_losses`` wrapper (which every paradigm passes
through) auto-emits a ``model_output`` snapshot from it. Diffusion's richer
pre/post-DC snapshot suppresses the generic one for that step via
``_model_output_snapshot_done``.

These exercise the new seams on a bare instance (the existing
strategy-unit-test pattern in ``test_base_debug_snapshot_kspace_2026_06``).
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch

import spectramr.infrastructure.training.debug_snapshot as ds_mod
import spectramr.infrastructure.training.utils.domain_inference as di_mod
from spectramr.infrastructure.training.strategies.base import (
    BaseTrainingStrategy,
    _first_tensor,
)


class _Strat(BaseTrainingStrategy):
    def _compute_losses_impl(self, *args, **kwargs):  # pragma: no cover - overridden per test
        return {"g_total_loss": torch.zeros(())}


def _bare(tmp_path) -> _Strat:
    s = object.__new__(_Strat)
    s.env = SimpleNamespace(run_output_dir=str(tmp_path), generator=None)
    s.config = SimpleNamespace(
        # The NESTED shape `_resolve_config` reads. `LoggingConfigSchema` folds
        # the retired flat spelling (`debug_snapshot_max_calls` → `snapshots
        # .max_calls`) in a validator, so real configs were never affected by
        # the block decomposition -- but a hand-built namespace never runs that
        # validator, so it must be written post-fold. Flat keys here resolve to
        # an `AttributeError` on `.snapshots`, which the emitter's blanket
        # handler swallows: the snapshot silently stops being written.
        logging=SimpleNamespace(
            snapshots=SimpleNamespace(
                enabled=True,
                max_calls=8,
                save_images=False,  # no torchvision needed
                save_json=True,
                interval_steps=0,
            )
        ),
        training=SimpleNamespace(output_dir=str(tmp_path)),
        # _compute_losses installs the dimension-contract guard, which reads
        # config.model.model_type (a declared field on the real frozen
        # TrainingSettings, so the source is right to read it directly).
        # get_model_capabilities returns None for an unannotated type, so the
        # guard installs no hook and these snapshot tests stay hermetic.
        model=SimpleNamespace(model_type="configurable_unet"),
    )
    return s


# ── _first_tensor: extract the primary tensor from a model output ──────────


def test_first_tensor_unwraps_tensor_tuple_dict() -> None:
    t = torch.ones(1, 1, 4, 4)
    assert _first_tensor(t) is not None and _first_tensor(t).shape == t.shape
    assert _first_tensor((t, "x")) is not None
    assert _first_tensor({"reconstruction": t, "aux": 3}).shape == t.shape
    assert _first_tensor({"output": t}).shape == t.shape
    assert _first_tensor("not a tensor") is None
    assert _first_tensor({"nope": 5}) is None


def test_first_tensor_detaches() -> None:
    t = torch.ones(1, 1, 4, 4, requires_grad=True) * 2
    out = _first_tensor(t)
    assert out is not None and not out.requires_grad  # detached for snapshotting


# ── forward hook capture ───────────────────────────────────────────────────


def test_forward_hook_captures_generator_output(tmp_path) -> None:
    s = _bare(tmp_path)
    gen = torch.nn.Conv2d(1, 1, 3, padding=1)
    s.env = SimpleNamespace(run_output_dir=str(tmp_path), generator=gen)
    s._ensure_generator_output_capture()
    # idempotent: a second call must NOT register a second hook
    s._ensure_generator_output_capture()

    s._capture_gen_output = True
    y = gen(torch.randn(2, 1, 8, 8))
    assert s._last_generator_output is not None
    assert s._last_generator_output.shape == y.shape
    assert not s._last_generator_output.requires_grad  # stashed detached


def test_forward_hook_noop_when_capture_disabled(tmp_path) -> None:
    s = _bare(tmp_path)
    gen = torch.nn.Conv2d(1, 1, 3, padding=1)
    s.env = SimpleNamespace(run_output_dir=str(tmp_path), generator=gen)
    s._ensure_generator_output_capture()
    s._capture_gen_output = False
    s._last_generator_output = None
    gen(torch.randn(1, 1, 8, 8))
    assert s._last_generator_output is None  # validation/witness forwards ignored


# ── _snapshot_model_output: the generic emission ───────────────────────────


def _patch_save(monkeypatch) -> dict:
    captured: dict = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(ds_mod, "save_debug_snapshot", _fake)
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    return captured


def test_snapshot_model_output_emits_and_clears(tmp_path, monkeypatch) -> None:
    captured = _patch_save(monkeypatch)
    s = _bare(tmp_path)
    s._last_generator_output = torch.ones(1, 1, 8, 8)
    s._model_output_snapshot_done = False
    s._snapshot_model_output(
        input_batch=torch.zeros(1, 1, 8, 8), target_batch=torch.zeros(1, 1, 8, 8), step=3
    )
    assert captured["tag"] == "model_output"
    assert set(captured["tensors"]) == {"model_output", "target", "input"}
    assert s._last_generator_output is None  # cleared so nothing is pinned across steps


def test_snapshot_model_output_suppressed_when_strategy_already_emitted(tmp_path, monkeypatch) -> None:
    """Diffusion emits its own richer model_output → the generic one must skip."""
    captured = _patch_save(monkeypatch)
    s = _bare(tmp_path)
    s._last_generator_output = torch.ones(1, 1, 8, 8)
    s._model_output_snapshot_done = True
    s._snapshot_model_output(
        input_batch=torch.zeros(1, 1, 8, 8), target_batch=torch.zeros(1, 1, 8, 8), step=0
    )
    assert captured == {}  # no emission
    assert s._last_generator_output is None  # still cleared


def test_snapshot_model_output_noop_without_capture(tmp_path, monkeypatch) -> None:
    captured = _patch_save(monkeypatch)
    s = _bare(tmp_path)
    s._last_generator_output = None
    s._snapshot_model_output(
        input_batch=torch.zeros(1, 1, 8, 8), target_batch=torch.zeros(1, 1, 8, 8), step=0
    )
    assert captured == {}


def test_model_output_pre_post_dc_are_authoritative_kspace(tmp_path, monkeypatch) -> None:
    """Diffusion's ``model_output_pre_dc`` / ``model_output_post_dc`` must get the
    authoritative unconditional IFFT (like ``target``/``input``), else the fragile
    spectrum veto renders them as raw k-space over the image tiles ("k-space on top
    of image space" — experiment_11 attention_none snapshot)."""
    captured: dict = {}
    monkeypatch.setattr(ds_mod, "save_debug_snapshot", lambda **kw: captured.update(kw))
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (True, True))
    s = _bare(tmp_path)
    t = torch.zeros(1, 8, 8, 8)
    s.save_debug_snapshot(
        {
            "model_output_backbone": t,
            "model_output_pre_dc": t,
            "model_output_post_dc": t,
            "target": t,
            "input": t,
        },
        step=0,
        tag="model_output",
        in_kspace_keys={
            "model_output_backbone",
            "model_output_pre_dc",
            "model_output_post_dc",
            "target",
            "input",
        },
    )
    auth = captured["authoritative_kspace_keys"]
    assert "model_output_backbone" in auth
    assert "model_output_pre_dc" in auth
    assert "model_output_post_dc" in auth


def test_save_debug_snapshot_sets_done_flag(tmp_path, monkeypatch) -> None:
    """A direct ``tag='model_output'`` call marks the step done (suppresses dup)."""
    _patch_save(monkeypatch)
    s = _bare(tmp_path)
    s._model_output_snapshot_done = False
    s.save_debug_snapshot(
        {"model_output": torch.zeros(1, 1, 8, 8)}, step=0, tag="model_output"
    )
    assert s._model_output_snapshot_done is True


# ── end-to-end through _compute_losses (real generator + hook + disk) ───────


def test_compute_losses_writes_model_output_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, False))
    gen = torch.nn.Conv2d(1, 1, 3, padding=1)
    s = _bare(tmp_path)
    s.env = SimpleNamespace(run_output_dir=str(tmp_path), generator=gen)
    s.amp_helper = SimpleNamespace(get_autocast_context=contextlib.nullcontext)

    def _impl(*, input_batch, target_batch, epoch, **kwargs):
        pred = gen(input_batch)
        return {"g_total_loss": torch.nn.functional.l1_loss(pred, target_batch)}

    s._compute_losses_impl = _impl  # type: ignore[method-assign]

    x = torch.randn(2, 1, 8, 8)
    y = torch.randn(2, 1, 8, 8)
    losses, _ = s._compute_losses(x, y, epoch=0, iteration=0)
    assert "g_total_loss" in losses
    snap = tmp_path / "debug_snapshots" / "model_output_step_000000"
    assert snap.is_dir(), "base _compute_losses must auto-emit a model_output snapshot"
    assert (snap / "snapshot.json").exists()
