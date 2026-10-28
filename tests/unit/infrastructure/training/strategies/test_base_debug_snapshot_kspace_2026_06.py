"""Tests for ``BaseTrainingStrategy.save_debug_snapshot``'s k-space key union
(2026-06-17 downloaded-run triage).

The wrapper must mark the canonical k-space tensor names (``target`` /
``input*`` / ``model_input``) as k-space whenever the dataloader yields k-space
(``needs_ifft_for_visualization`` → ``targets_are_kspace=True``), so the
previewer IFFTs them into human-readable images. Two paths regressed:

* the ``first_steps`` auto-snapshot passes ``in_kspace_keys=None`` → must mark
  every key;
* the diffusion ``model_output`` snapshot passes an EXPLICIT set that omitted
  ``"target"`` → ``target.png`` rendered as raw k-space. The wrapper must UNION
  the canonical names into the caller's set, not leave it as-is.

The method touches no construction state, so it is exercised on a bare instance
(the existing strategy-unit-test pattern).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

import spectramr.infrastructure.training.debug_snapshot as ds_mod
import spectramr.infrastructure.training.utils.domain_inference as di_mod
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy


class _Strat(BaseTrainingStrategy):
    def _compute_losses_impl(self, *args, **kwargs):  # pragma: no cover - stub
        return {}


def _bare(tmp_path) -> _Strat:
    s = object.__new__(_Strat)
    s.env = SimpleNamespace(run_output_dir=str(tmp_path))
    # config only needs a ``logging`` attr here — needs_ifft is monkeypatched.
    s.config = SimpleNamespace(
        # Nested shape, as `_resolve_config` reads it. The flat spelling is
        # folded by a `LoggingConfigSchema` validator, which a hand-built
        # namespace never runs -- so it must be written post-fold here.
        logging=SimpleNamespace(
            snapshots=SimpleNamespace(
                enabled=True,
                max_calls=8,
                save_images=False,  # no file I/O needed
                save_json=False,
                interval_steps=0,
            )
        ),
        training=SimpleNamespace(output_dir=str(tmp_path)),
    )
    return s


def _capture(monkeypatch):
    """Patch the module-level save_debug_snapshot; return a dict that records
    the ``in_kspace_keys`` the base wrapper forwarded."""
    captured: dict = {}

    def _fake(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(ds_mod, "save_debug_snapshot", _fake)
    return captured


def _set_targets_kspace(monkeypatch, value: bool) -> None:
    monkeypatch.setattr(
        di_mod, "needs_ifft_for_visualization", lambda _cfg: (False, value)
    )


def test_explicit_keys_get_target_unioned(tmp_path, monkeypatch) -> None:
    """The model_output regression: caller omits 'target' → wrapper adds it."""
    _set_targets_kspace(monkeypatch, True)
    captured = _capture(monkeypatch)
    s = _bare(tmp_path)
    s.save_debug_snapshot(
        {"target": torch.zeros(1, 2, 8, 8), "model_output_post_dc": torch.zeros(1, 2, 8, 8)},
        step=0,
        tag="model_output",
        in_kspace_keys={"model_output_post_dc"},
    )
    assert "target" in captured["in_kspace_keys"]
    assert "model_output_post_dc" in captured["in_kspace_keys"]  # caller's key preserved


def test_none_keys_marks_all_when_targets_kspace(tmp_path, monkeypatch) -> None:
    """The first_steps path: in_kspace_keys=None → every key marked."""
    _set_targets_kspace(monkeypatch, True)
    captured = _capture(monkeypatch)
    s = _bare(tmp_path)
    tensors = {"input_raw": torch.zeros(1, 2, 8, 8), "target": torch.zeros(1, 2, 8, 8)}
    s.save_debug_snapshot(tensors, step=0, tag="first_steps")
    assert captured["in_kspace_keys"] == set(tensors.keys())


def test_image_dataset_does_not_union(tmp_path, monkeypatch) -> None:
    """When the dataloader yields IMAGE targets, the wrapper must NOT force
    canonical keys into the set (no spurious IFFT flag)."""
    _set_targets_kspace(monkeypatch, False)
    captured = _capture(monkeypatch)
    s = _bare(tmp_path)
    s.save_debug_snapshot(
        {"target": torch.zeros(1, 1, 8, 8)},
        step=0,
        tag="first_steps",
        in_kspace_keys={"prediction"},
    )
    # left exactly as the caller passed it (no "target" injected)
    assert captured["in_kspace_keys"] == {"prediction"}


def test_canonical_keys_are_authoritative_when_targets_kspace(tmp_path, monkeypatch) -> None:
    """The previewer must IFFT config-declared k-space tensors unconditionally
    (2026-06-27 audit). The wrapper forwards the canonical k-space names present
    as ``authoritative_kspace_keys`` so the previewer bypasses the fragile
    spectrum veto that false-negated on real normalized M4Raw k-space."""
    _set_targets_kspace(monkeypatch, True)
    captured = _capture(monkeypatch)
    s = _bare(tmp_path)
    s.save_debug_snapshot(
        {
            "input_raw": torch.zeros(1, 2, 8, 8),
            "input_prepared": torch.zeros(1, 2, 8, 8),
            "target": torch.zeros(1, 2, 8, 8),
        },
        step=0,
        tag="first_steps",
    )
    assert captured["authoritative_kspace_keys"] == {
        "input_raw", "input_prepared", "target",
    }


def test_explicit_keys_target_is_authoritative(tmp_path, monkeypatch) -> None:
    """The model_output path: ``target`` and the diffusion ``model_output_post_dc``
    are both canonical k-space → authoritative (unconditional IFFT). Leaving
    ``model_output_post_dc`` to the fragile veto rendered it as raw k-space over
    the image tiles ("k-space on top of image space", experiment_11 attention_none)."""
    _set_targets_kspace(monkeypatch, True)
    captured = _capture(monkeypatch)
    s = _bare(tmp_path)
    s.save_debug_snapshot(
        {"target": torch.zeros(1, 2, 8, 8), "model_output_post_dc": torch.zeros(1, 2, 8, 8)},
        step=0,
        tag="model_output",
        in_kspace_keys={"model_output_post_dc"},
    )
    assert captured["authoritative_kspace_keys"] == {"target", "model_output_post_dc"}


def test_image_dataset_has_no_authoritative_keys(tmp_path, monkeypatch) -> None:
    """Image-output runs (VF / IB) declare no k-space domain → the veto still
    guards every key (no unconditional IFFT)."""
    _set_targets_kspace(monkeypatch, False)
    captured = _capture(monkeypatch)
    s = _bare(tmp_path)
    s.save_debug_snapshot(
        {"target": torch.zeros(1, 1, 8, 8)},
        step=0,
        tag="first_steps",
        in_kspace_keys={"prediction"},
    )
    assert captured["authoritative_kspace_keys"] == set()


# ── domain_exact: an image-domain model_input on a k-space loader (#3043) ──


def _image_2ch() -> torch.Tensor:
    """A smooth 2-channel (real, imag) image, as ``A^H y`` hands an image network."""
    yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    img = torch.exp(-(((yy - 32) ** 2 + (xx - 32) ** 2) / 300)) * (1 + 0.3 * torch.sin(xx / 3))
    return torch.stack([img, 0.2 * img]).unsqueeze(0)


def test_domain_exact_skips_the_canonical_union(tmp_path, monkeypatch) -> None:
    """An image network's ``model_input`` is not k-space even though the loader is."""
    _set_targets_kspace(monkeypatch, True)
    captured = _capture(monkeypatch)
    _bare(tmp_path).save_debug_snapshot(
        {"model_input": _image_2ch()},
        step=0,
        tag="ei_model_input",
        in_kspace_keys=set(),
        domain_exact=True,
    )
    assert captured["in_kspace_keys"] == set()
    assert captured["authoritative_kspace_keys"] == set()


def test_empty_set_without_domain_exact_still_unions_model_input(tmp_path, monkeypatch) -> None:
    """EDM, flow matching, stochastic interpolants and the VF twin pass ``set()`` and
    rely on the union: ``x_t`` lives in the target's domain."""
    _set_targets_kspace(monkeypatch, True)
    captured = _capture(monkeypatch)
    _bare(tmp_path).save_debug_snapshot(
        {"model_input": torch.zeros(1, 2, 8, 8)},
        step=0,
        tag="diffusion_step",
        in_kspace_keys=set(),
    )
    assert "model_input" in captured["in_kspace_keys"]
    assert "model_input" in captured["authoritative_kspace_keys"]


def test_declared_domain_exact_reaches_the_wrapper_and_is_consumed(tmp_path, monkeypatch) -> None:
    _set_targets_kspace(monkeypatch, True)
    captured = _capture(monkeypatch)
    s = _bare(tmp_path)
    s.snapshot_prepared_is_model_input = False
    s.snapshot_model_input_tag = "ei_model_input"
    s._declared_model_input = None
    s._model_input_snapshot_done = False
    s._model_input_width = (2, "test")
    s._model_input_contract_warned = False
    s._declare_model_input(
        {"model_input": _image_2ch()},
        extra={"model_input_key": "model_input"},
        in_kspace_keys=set(),
        domain_exact=True,
    )
    s._snapshot_declared_model_input(step=1)
    assert captured["authoritative_kspace_keys"] == set()
    assert s._declared_domain_exact is False  # one step's declaration never leaks into the next


def test_image_model_input_renders_as_an_image_not_a_dot() -> None:
    """The planted violation: the union's output (authoritative IFFT of an image) is a
    single pixel; the domain-exact output is the image."""
    x = _image_2ch()
    broken = ds_mod._render_image_preview(x, in_kspace=True, authoritative=True)
    fixed = ds_mod._render_image_preview(x, in_kspace=False)
    assert (broken > 0.5).float().mean() < 0.01
    assert (fixed > 0.5).float().mean() > 0.10
