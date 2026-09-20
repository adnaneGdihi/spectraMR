"""Wiring pins for the t=0 pre-DC probe in `DiffusionTrainingStrategy`.

Non-negotiable 16: registering a capability is the easy half. These execute the
real method body against a stub `self`, so "the probe exists" and "the probe
runs" are different assertions and only the second one passes here.

The structural pin at the bottom exists because a name-presence check cannot
tell a GUARD from a call: `emit_reports` appears in the source either way. The
guard is what keeps the probe's scoring pass from writing a second set of
TensorBoard renders and a second report-recorder row per batch.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

import torch

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)


class _Generator:
    exposes_pre_dc = True

    def __init__(self):
        self.calls = []

    def __call__(self, x, timesteps=None, *, return_pre_dc=False, **kwargs):
        self.calls.append({"timesteps": timesteps, "kwargs": kwargs})
        assert return_pre_dc, "the strategy must ask for the pre-DC proposal"
        return x, torch.full_like(x, 7.0)


def _make_self(generator, *, metrics=None, chunk_size=2):
    """A stub `self` carrying only what `_t0_pre_dc_probe_metrics` reaches for."""
    recorded = {"score_calls": []}

    def _build_generator_kwargs(**kw):
        recorded["build_kwargs"] = kw
        return {"smaps": torch.ones(1, 2, 4, 4), "mask": kw["mask"]}

    def _compute_validation_metrics(*args, **kwargs):
        recorded["score_calls"].append({"args": args, "kwargs": kwargs})
        return dict(metrics or {"val_psnr": 21.0})

    obj = SimpleNamespace(
        generator_model=generator,
        _is_cold_diffusion=lambda: True,
        _build_generator_kwargs=_build_generator_kwargs,
        _compute_validation_metrics=_compute_validation_metrics,
        # The probe chunks its forward on the same knob as the cascade rungs,
        # so the stub has to carry it; `validation: None` is a legal arm and
        # reaches this method too, which the sibling test below pins.
        config=SimpleNamespace(
            validation=SimpleNamespace(loader=SimpleNamespace(chunk_size=chunk_size))
        ),
    )
    return obj, recorded


def _run(obj, *, batch=1):
    return DiffusionTrainingStrategy._t0_pre_dc_probe_metrics(
        obj,
        target_batch=torch.zeros(batch, 2, 4, 4),
        input_batch=torch.zeros(batch, 2, 4, 4),
        batch_data={},
        scale_factor=torch.ones(1),
        batch_idx=0,
    )


def test_probe_emits_namespaced_metrics():
    gen = _Generator()
    obj, _ = _make_self(gen)
    assert _run(obj) == {"val_t0_predc_psnr": 21.0}


def test_probe_actually_calls_the_generator_at_timestep_zero():
    gen = _Generator()
    obj, _ = _make_self(gen)
    _run(obj)
    assert len(gen.calls) == 1
    assert torch.equal(gen.calls[0]["timesteps"], torch.zeros(1, dtype=torch.long))


def test_probe_scores_the_pre_dc_tensor_not_the_post_dc_one():
    """The stub returns `x` post-DC and 7.0 pre-DC; scoring must see 7.0."""
    gen = _Generator()
    obj, rec = _make_self(gen)
    _run(obj)
    scored = rec["score_calls"][0]["args"][0]
    assert torch.equal(scored, torch.full((1, 2, 4, 4), 7.0))


def test_probe_suppresses_reports_so_it_cannot_overwrite_cascade_renders():
    """The spy the image-overwrite regression deserves.

    `_compute_validation_metrics` also feeds `feed_report_case_recorder`, so a
    probe that let reports through would add a row per batch AND, at
    `cascade_level=None`, take the legacy single-prefix path that overwrote the
    cascade images ("experiment_11 fake images doubled").
    """
    gen = _Generator()
    obj, rec = _make_self(gen)
    _run(obj)
    assert rec["score_calls"][0]["kwargs"]["emit_reports"] is False


def test_probe_passes_an_all_ones_mask_because_t0_reveals_everything():
    gen = _Generator()
    obj, rec = _make_self(gen)
    _run(obj)
    mask = rec["build_kwargs"]["mask"]
    assert torch.equal(mask, torch.ones_like(mask))
    assert mask.shape[1] == 1


def test_ddp_wrapped_generator_is_unwrapped():
    """A wrapper has no `exposes_pre_dc`, so the probe would go silently empty.

    Consistent across ranks, so no hang -- just no readout, which is the
    failure mode this whole change exists to remove.
    """
    gen = _Generator()
    wrapped = SimpleNamespace(module=gen)
    obj, _ = _make_self(wrapped)
    assert _run(obj) == {"val_t0_predc_psnr": 21.0}
    assert len(gen.calls) == 1


def test_generator_without_the_capability_emits_nothing_and_never_scores():
    class Plain:
        def __call__(self, *a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("probe called a generator with no pre-DC")

    obj, rec = _make_self(Plain())
    assert _run(obj) == {}
    assert rec["score_calls"] == []


def test_probe_chunks_its_forward_on_validation_loader_chunk_size():
    """The knob the cascade rungs honour reaches the probe too.

    An unchunked pass here is the densest forward of the sweep -- the depth
    flatten at an all-ones mask, run after the cascade has filled the card --
    and it was the one validation forward that ignored the knob.
    """
    gen = _Generator()
    obj, _ = _make_self(gen, chunk_size=2)
    _run(obj, batch=6)
    assert [int(c["timesteps"].shape[0]) for c in gen.calls] == [2, 2, 2]


def test_an_arm_with_no_validation_block_chunks_at_one_row():
    """`config.validation` is Optional and defaults to None (a legal arm).

    Dereferencing it here would turn a memory guard into an AttributeError on
    the arms least likely to have tuned anything.
    """
    gen = _Generator()
    obj, _ = _make_self(gen)
    obj.config = SimpleNamespace(validation=None)
    assert _run(obj, batch=3) == {"val_t0_predc_psnr": 21.0}
    assert [int(c["timesteps"].shape[0]) for c in gen.calls] == [1, 1, 1]


# --- Detector cores (pure, so they can be planted) ---------------------------
#
# Extracted from the two pins below because a detector that has never been
# shown a violation is not a gate (non-negotiable 15). The plants live in
# `test_diffusion_probe_pin_detectors_2026_09.py`, which imports these.


def parse_method_source(source: str) -> ast.Module:
    """Parse a method's own source, de-indented.

    ``textwrap.dedent`` rather than ``.replace("\n    ", "\n")``: the replace
    hack also eats four spaces from continuation lines inside multi-line string
    literals, silently rewriting the very source the pin is reading.
    """
    return ast.parse(textwrap.dedent(source), mode="exec")


def _attr_calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == name
    ]


def image_logger_call_counts(source: str, *, guard: str = "emit_reports") -> tuple[int, int]:
    """``(total, guarded)`` calls to the validation image logger in ``source``.

    A call counts as guarded only under an ``if <guard>:`` whose test is a bare
    ``Name``. ``if emit_reports and x:`` is a ``BoolOp`` and is deliberately
    counted UNGUARDED -- a compound condition can be false for reasons the
    probe does not control, so the pin refuses it rather than guessing.
    """
    tree = parse_method_source(source)
    target = "_log_validation_images_to_tensorboard"
    total = _attr_calls(tree, target)
    guarded: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not (isinstance(node.test, ast.Name) and node.test.id == guard):
            continue
        guarded.extend(_attr_calls(node, target))
    return len(total), len(guarded)


def probe_call_site_count(source: str, *, name: str = "_t0_pre_dc_probe_metrics") -> int:
    """How many times ``source`` CALLS ``self.<name>``.

    Attribute calls only: a bare ``name(...)`` is not counted, because the
    production shape is always ``self.<name>(...)`` and widening the match
    would let an unrelated local function satisfy the pin.
    """
    return len(_attr_calls(parse_method_source(source), name))


# --- The pins ----------------------------------------------------------------


def test_validation_step_wires_the_probe_on_every_return_path():
    """Both returns must carry it, or the readout depends on `config.undersampling`."""
    n = probe_call_site_count(inspect.getsource(DiffusionTrainingStrategy.validation_step))
    assert n == 2, f"expected the probe on both return paths, found {n}"


def test_the_image_logger_call_is_guarded_by_emit_reports():
    """Structural, not substring: a mention of `emit_reports` is not a guard.

    Pins that the ONE side-effecting call inside the metrics seam sits under an
    `if emit_reports`, so `emit_reports=False` genuinely suppresses it rather
    than merely being accepted and ignored.
    """
    total, guarded = image_logger_call_counts(
        inspect.getsource(DiffusionTrainingStrategy._compute_validation_metrics)
    )
    assert guarded, (
        "_log_validation_images_to_tensorboard is not inside an `if emit_reports` "
        "block; the probe's scoring pass would write renders and report rows"
    )
    assert total == guarded, "an unguarded call to the image logger remains in the metrics seam"


def test_emit_reports_defaults_to_true_so_the_cascade_is_unaffected():
    param = inspect.signature(DiffusionTrainingStrategy._compute_validation_metrics).parameters[
        "emit_reports"
    ]
    assert param.default is True
