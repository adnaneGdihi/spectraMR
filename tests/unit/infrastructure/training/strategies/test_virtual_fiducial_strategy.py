"""Domain-routing regression tests for ConcreteVirtualFiducialStrategy.

Guards the fix where ``losses.complex_losses`` were silently fed magnitude
tensors, zeroing phase supervision for the phase-physics VF arms (B0 / B1 /
phase-navigator) whose primary metric is ``val_phase_mse``.

These exercise the pure static router ``_apply_domain_losses`` directly so we
avoid the heavyweight BaseTrainingStrategy DI/AMP init chain (same convention
as ``tests/unit/infrastructure/training/test_vf_strategy_val_metrics.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.virtual_fiducial_strategy import (
    ConcreteVirtualFiducialStrategy,
)


class _SpyLoss(nn.Module):
    """Records whether the prediction tensor it received was complex."""

    def __init__(self) -> None:
        super().__init__()
        self.saw_complex: bool | None = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.saw_complex = bool(torch.is_complex(pred))
        if torch.is_complex(pred):
            diff = pred - target
            return (diff.real**2 + diff.imag**2).mean()
        return ((pred - target) ** 2).mean()


def _complex_pair(phase_shift: float = 0.0):
    """Two complex tensors with IDENTICAL magnitude, differing only in phase."""
    mag = torch.rand(2, 1, 8, 8) + 0.5
    target = mag.to(torch.complex64)  # phase 0
    pred = (mag * torch.exp(torch.tensor(1j * phase_shift))).to(torch.complex64)
    return pred, target


def test_complex_loss_receives_complex_tensor():
    spy = _SpyLoss()
    pred, target = _complex_pair()
    ConcreteVirtualFiducialStrategy._apply_domain_losses(
        [("spy", spy, 1.0, "complex")], pred, target, torch.device("cpu")
    )
    assert spy.saw_complex is True


def test_image_loss_receives_magnitude_tensor():
    spy = _SpyLoss()
    pred, target = _complex_pair()
    ConcreteVirtualFiducialStrategy._apply_domain_losses(
        [("spy", spy, 1.0, "image")], pred, target, torch.device("cpu")
    )
    assert spy.saw_complex is False


def test_kspace_loss_receives_complex_kspace():
    spy = _SpyLoss()
    pred, target = _complex_pair()
    ConcreteVirtualFiducialStrategy._apply_domain_losses(
        [("spy", spy, 1.0, "kspace")], pred, target, torch.device("cpu")
    )
    assert spy.saw_complex is True


def test_phase_only_difference_supervised_by_complex_not_image():
    """The regression: a pure phase shift is invisible to the magnitude loss
    but caught by the complex loss. The old all-magnitude path missed it."""
    pred, target = _complex_pair(phase_shift=0.5)

    img_total, _ = ConcreteVirtualFiducialStrategy._apply_domain_losses(
        [("spy", _SpyLoss(), 1.0, "image")], pred, target, torch.device("cpu")
    )
    cplx_total, _ = ConcreteVirtualFiducialStrategy._apply_domain_losses(
        [("spy", _SpyLoss(), 1.0, "complex")], pred, target, torch.device("cpu")
    )

    assert img_total.item() < 1e-6, "magnitude loss should be blind to phase"
    assert cplx_total.item() > 1e-3, "complex loss must penalize phase error"


def test_weights_and_loss_dict_keys():
    spy_a, spy_b = _SpyLoss(), _SpyLoss()
    pred, target = _complex_pair(phase_shift=0.3)
    total, loss_dict = ConcreteVirtualFiducialStrategy._apply_domain_losses(
        [("a", spy_a, 2.0, "complex"), ("b", spy_b, 0.5, "image")],
        pred,
        target,
        torch.device("cpu"),
    )
    assert "loss_a" in loss_dict and "loss_b" in loss_dict
    expected = 2.0 * loss_dict["loss_a"] + 0.5 * loss_dict["loss_b"]
    assert torch.allclose(total, expected)


# ── deformation-field routing (m5/m9 hyperelastic-Jacobian) ───────────────────


class _DeformSpyLoss(nn.Module):
    """A loss (like HyperelasticJacobianLoss) that regularises the deformation
    field passed via ``intermediate_outputs`` — not the prediction image."""

    def __init__(self) -> None:
        super().__init__()
        self.saw_field: torch.Tensor | None = None

    def forward(self, pred, target=None, intermediate_outputs=None):
        if intermediate_outputs:
            self.saw_field = intermediate_outputs[0]
            return intermediate_outputs[0].abs().mean()
        return pred.abs().mean()


def test_intermediate_outputs_routed_to_deformation_loss():
    """A loss whose signature accepts intermediate_outputs must receive the
    deformation field — so the incompressibility constraint acts on the VELOCITY
    field, not the reconstructed image (m5/m9 regression)."""
    spy = _DeformSpyLoss()
    pred, target = _complex_pair()
    field = torch.randn(2, 2, 8, 8)
    ConcreteVirtualFiducialStrategy._apply_domain_losses(
        [("hyperelastic_jacobian", spy, 1.0, "image")],
        pred,
        target,
        torch.device("cpu"),
        intermediate_outputs=[field],
    )
    assert spy.saw_field is not None, "deformation field was not routed to the loss"
    assert torch.equal(spy.saw_field, field)


def test_intermediate_outputs_not_passed_to_plain_loss():
    """A plain loss whose signature does NOT accept intermediate_outputs must not
    receive it (no TypeError) — the routing is signature-gated."""
    spy = _SpyLoss()  # forward(pred, target) only
    pred, target = _complex_pair()
    field = torch.randn(2, 2, 8, 8)
    total, _ = ConcreteVirtualFiducialStrategy._apply_domain_losses(
        [("l1", spy, 1.0, "image")],
        pred,
        target,
        torch.device("cpu"),
        intermediate_outputs=[field],
    )
    assert spy.saw_complex is False  # ran normally, no crash


# ---------------------------------------------------------------------------
# Adaptive-conditioning wiring (2026-05-26). Static/pure helpers + the
# optimizer-registration instance method are exercised via a fake ``self`` to
# avoid the heavyweight BaseTrainingStrategy DI/AMP init (same convention as
# the domain-loss router tests above).
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from spectramr.config.schemas.conditioning import ConditioningConfig  # noqa: E402
from spectramr.models.conditioning import (  # noqa: E402
    AdaptiveConditioner,
    ConditioningContext,
)

_VF = ConcreteVirtualFiducialStrategy


def _fake_sim(**kw):
    base = dict(
        _motion_severity=2.0,
        b0_strength=0.3,
        enable_b0=True,
        b1_strength=0.2,
        enable_b1=True,
        snr_range=(10.0, 25.0),
        acceleration=4.0,
        enable_undersampling=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_severity_vector_shape_and_values():
    vec = _VF._severity_vector(_fake_sim(), batch_size=3, device=torch.device("cpu"))
    assert vec.shape == (3, 5)
    assert vec[0, 0].item() == pytest.approx(2.0)  # motion_severity
    assert vec[0, 1].item() == pytest.approx(0.3)  # b0
    assert vec[0, 2].item() == pytest.approx(0.2)  # b1


def test_severity_vector_zeros_disabled_fields():
    vec = _VF._severity_vector(_fake_sim(enable_b0=False, enable_b1=False), 1, torch.device("cpu"))
    assert vec[0, 1].item() == 0.0 and vec[0, 2].item() == 0.0


def test_build_context_populates_severity():
    ctx = _VF._build_conditioning_context(["severity_vec"], _fake_sim(), 2, torch.device("cpu"))
    assert isinstance(ctx, ConditioningContext) and ctx.has("severity_vec")


def test_build_context_none_when_no_sources():
    assert _VF._build_conditioning_context([], _fake_sim(), 2, torch.device("cpu")) is None


def test_build_context_rejects_unsupported_source():
    with pytest.raises(ValueError, match="diffusion_t"):
        _VF._build_conditioning_context(["diffusion_t"], _fake_sim(), 2, torch.device("cpu"))


def test_ensure_conditioner_registers_params_with_optimizer():
    cfg = ConditioningConfig(enabled=True, sources=["severity_vec"], embed_dim=8)
    dummy = nn.Linear(2, 2)
    opt = torch.optim.Adam(dummy.parameters(), lr=1e-3)
    fake = SimpleNamespace(
        config=SimpleNamespace(model=SimpleNamespace(conditioning=cfg)),
        env=SimpleNamespace(opt_g=opt),
        device=torch.device("cpu"),
        _conditioner=None,
    )
    cond = _VF._ensure_conditioner(fake, n_channels=2)
    assert cond is not None
    assert len(opt.param_groups) == 2  # conditioner params registered for training


def test_ensure_conditioner_disabled_returns_none():
    fake = SimpleNamespace(
        config=SimpleNamespace(model=SimpleNamespace(conditioning=ConditioningConfig())),
        env=SimpleNamespace(opt_g=None),
        device=torch.device("cpu"),
        _conditioner=None,
    )
    assert _VF._ensure_conditioner(fake, 2) is None


def test_conditioner_film_on_input_is_identity_at_init():
    cfg = ConditioningConfig(enabled=True, sources=["severity_vec"], embed_dim=8)
    cond = AdaptiveConditioner.from_config(cfg, num_features=2)
    ctx = _VF._build_conditioning_context(["severity_vec"], _fake_sim(), 2, torch.device("cpu"))
    x = torch.randn(2, 2, 8, 8)
    assert torch.allclose(cond(x, ctx), x, atol=1e-6)


# ---------------------------------------------------------------------------
# F1 (smoke_audit_20260526): validation_step rendered the k-space prediction
# as the "fake" image without an IFFT, so every ``output_domain: kspace`` VF
# arm produced a 4-corner-blob / concentric-ring artifact (|k-space| shown as
# an image) and a cross-domain ``val_psnr``. The fix routes the prediction
# through ``_to_image_domain`` keyed on the authoritative ``infer_output_domain``,
# symmetric to the target IFFT already present in ``validation_step``.
# ---------------------------------------------------------------------------

from spectramr.infrastructure.physics.fft_ops import FFTTransformer  # noqa: E402


def test_kspace_prediction_is_ifft_to_image_before_visual():
    """A k-space prediction must be IFFT'd to image domain for the visual.

    Regression: builds k-space whose IFFT is a single bright centre pixel; the
    helper must recover that image, NOT pass the raw k-space through.
    """
    device = torch.device("cpu")
    image = torch.zeros(1, 1, 16, 16, dtype=torch.complex64)
    image[0, 0, 8, 8] = 1.0
    ksp = FFTTransformer(device=device).fft2c(image)

    out = _VF._to_image_domain(ksp, "kspace", device)

    assert torch.allclose(out, image, atol=1e-5), "k-space pred must be IFFT'd to image"
    # The bug rendered |k-space| directly — guard against that exact regression.
    assert not torch.allclose(out.abs(), ksp.abs(), atol=1e-3)


def test_image_prediction_passes_through_unchanged():
    """An image-domain prediction must NOT be transformed (no double-IFFT)."""
    device = torch.device("cpu")
    image = torch.rand(1, 1, 16, 16, dtype=torch.complex64)
    out = _VF._to_image_domain(image, "image", device)
    assert torch.allclose(out, image)


# ── 2026-06-02: measurement (mask) is threaded into the DC-bearing model ──


def test_forward_model_threads_undersampling_mask() -> None:
    """_forward_model must add the twin's sampling mask to physics_kwargs."""
    import inspect

    from spectramr.infrastructure.training.strategies.virtual_fiducial_strategy import (
        ConcreteVirtualFiducialStrategy,
    )

    src = inspect.getsource(ConcreteVirtualFiducialStrategy._forward_model)
    assert "undersampling_mask_kwargs(self.simulator)" in src


# ── VF marker routing to marker_signal models (VF review 2026-06-04) ──────────
# virtual_fiducial_strategy._forward_model used to route the marker only to
# generators with a cross_attention/bridge attribute; NeuralAdvectionGenerator
# and DynamicMRNeRF take a per-sample `marker_signal` vector but expose neither,
# so the marker was silently DROPPED and the VF mechanism was inert (m5m9/m10
# produced time-invariant / identity output). These pin the detection seam, the
# scalar derivation, and the end-to-end routing.

_VFS = ConcreteVirtualFiducialStrategy


class _MarkerSignalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_marker_signal = None

    def forward(self, x, marker_signal, **kwargs):
        self.seen_marker_signal = marker_signal
        return x  # echo real-stacked tensor


class _XAttnModel(nn.Module):
    cross_attention = True

    def __init__(self) -> None:
        super().__init__()
        self.seen_residual = None

    def forward(self, x, marker_residual, **kwargs):
        self.seen_residual = marker_residual
        return x


class _PlainModel(nn.Module):
    def forward(self, x, **kwargs):
        return x


def test_forward_accepts_marker_signal_detection() -> None:
    assert _VFS._forward_accepts_marker_signal(_MarkerSignalModel()) is True
    assert _VFS._forward_accepts_marker_signal(_XAttnModel()) is False
    assert _VFS._forward_accepts_marker_signal(_PlainModel()) is False


def test_marker_signal_from_residual_is_per_sample_scalar() -> None:
    res = torch.randn(3, 2, 8, 8, dtype=torch.complex64)
    sig = _VFS._marker_signal_from_residual(res)
    assert sig.shape == (3, 1)
    assert torch.isfinite(sig).all()
    assert (sig >= 0).all()  # mean magnitude is non-negative
    # Larger-magnitude residual ⇒ larger signal (it tracks corruption strength).
    big = _VFS._marker_signal_from_residual(res * 5.0)
    assert (big > sig - 1e-6).all()


def _bare_vfs(generator) -> ConcreteVirtualFiducialStrategy:
    s = ConcreteVirtualFiducialStrategy.__new__(ConcreteVirtualFiducialStrategy)
    s.env = SimpleNamespace(generator=generator)
    s._conditioning_context = None
    s._ensure_conditioner = lambda n: None  # skip FiLM conditioner
    s.simulator = SimpleNamespace(last_undersampling_mask=None)  # → no mask kwarg
    return s


def test_forward_model_routes_marker_signal_to_neural_advection_like_model() -> None:
    gen = _MarkerSignalModel()
    strat = _bare_vfs(gen)
    b = 2
    corrupted = torch.randn(b, 1, 8, 8, dtype=torch.complex64)
    marker_res = torch.randn(b, 1, 8, 8, dtype=torch.complex64)
    strat._forward_model(corrupted, marker_res)
    assert gen.seen_marker_signal is not None, "marker_signal was dropped (the bug)"
    assert gen.seen_marker_signal.shape == (b, 1)


def test_forward_model_still_routes_residual_to_cross_attention_model() -> None:
    gen = _XAttnModel()
    strat = _bare_vfs(gen)
    b = 2
    corrupted = torch.randn(b, 1, 8, 8, dtype=torch.complex64)
    marker_res = torch.randn(b, 1, 8, 8, dtype=torch.complex64)
    strat._forward_model(corrupted, marker_res)
    assert gen.seen_residual is not None  # cross-attention path unchanged
    assert gen.seen_residual.shape[1] == 2  # 2-channel real-stacked marker image


# ── 2026-06-07: twin-output debug snapshot (marker-firing visibility) ─────────
# The base "first_steps" snapshot captures the PRE-twin input, so input_raw ==
# input_prepared == target and it cannot confirm the DigitalTwinSimulator fired
# (pitfall #16 facade risk). `_snapshot_twin_outputs` captures the twin OUTPUTS
# and stamps a marker_mechanism_fired flag. These pin the capture contract.


def _twin_snapshot_self(tmp_path, max_calls: int = 8):
    """A bare VFS wired to the REAL ``save_debug_snapshot``, writing under tmp_path.

    Deliberately NOT a spy. The previous version of this helper replaced
    ``self.save_debug_snapshot`` with a recorder, which made every assertion
    below a statement about the arguments rather than about the artifact -- and
    the arguments are exactly where #1188 hides: ``extra`` is now a zero-arg
    callable, so a spy sees a function object where a reader sees the values
    only after the writer resolves it. A recorder goes green on a ``vf_twin``
    snapshot that never reached disk.

    It also configured ``logging.debug_snapshot_max_calls``, retired by RENAMES
    to ``logging.snapshots.max_calls`` -- so it set nothing, and the budget it
    named was never the budget under test. The real schema stub raises on an
    unknown leaf instead.
    """
    from tests.utils.block_config_stub import LoggingConfigStub

    s = ConcreteVirtualFiducialStrategy.__new__(ConcreteVirtualFiducialStrategy)
    s.env = SimpleNamespace(run_output_dir=str(tmp_path))
    s.config = SimpleNamespace(
        training=SimpleNamespace(output_dir=str(tmp_path)),
        logging=LoggingConfigStub(snapshots={"max_calls": max_calls}),
        data=None,
    )
    s._model_output_snapshot_done = False
    # `vf_twin` IS this class's `snapshot_model_input_tag`, so the wrapper runs
    # the model-input contract on it (#1298) and reads the state `__init__`
    # sets. A bare-instance fixture has to mirror the constructor, and the cost
    # of not doing so is invisible here: `_snapshot_twin_outputs` swallows every
    # exception, so a missing attribute reads as "no snapshot was written".
    s._model_input_snapshot_done = False
    s._model_input_width = None
    s._model_input_contract_warned = False
    return s


def _twin_extra(tmp_path, step: int) -> dict:
    """The ``extra`` block of the vf_twin snapshot actually written for ``step``."""
    import json

    snap = tmp_path / "debug_snapshots" / f"vf_twin_step_{step:06d}" / "snapshot.json"
    return json.loads(snap.read_text())["extra"]


class _SyncCountingTensor:
    """A ``marker_residual`` stand-in counting the device sync it would cost.

    ``.detach()`` is the first hop of ``float(t.detach().abs().max().item())``,
    so one call here stands for one device->host sync -- the thing
    non-negotiable 9 forbids in the training loop.
    """

    def __init__(self, real: torch.Tensor) -> None:
        self._real = real
        self.syncs = 0

    def detach(self) -> torch.Tensor:
        self.syncs += 1
        return self._real

    def __getattr__(self, name: str):
        return getattr(self._real, name)


def _cplx(*shape):
    return torch.rand(*shape, dtype=torch.complex64)


def test_twin_snapshot_captures_outputs_and_flags_firing(tmp_path) -> None:
    s = _twin_snapshot_self(tmp_path)
    marker_res = _cplx(2, 1, 8, 8) + 0.1  # non-zero ⇒ mechanism fired
    s._snapshot_twin_outputs(
        step=1,
        epoch=0,
        clean_norm=_cplx(2, 1, 8, 8),
        corrupted_norm=_cplx(2, 1, 8, 8),
        marker_residual=marker_res,
        marker_prior_norm=_cplx(2, 1, 8, 8),
    )
    snap_dir = tmp_path / "debug_snapshots" / "vf_twin_step_000001"
    assert snap_dir.is_dir()
    report = (snap_dir / "snapshot.txt").read_text()
    for key in (
        "twin_target_clean",
        "twin_corrupted_input",
        "twin_marker_residual",
        "twin_marker_prior",
    ):
        assert key in report
    extra = _twin_extra(tmp_path, 1)
    assert extra["marker_mechanism_fired"] is True
    assert extra["marker_residual_abs_max"] > 0.0


def test_twin_snapshot_flags_inert_marker_when_residual_zero(tmp_path) -> None:
    """A zero marker residual (inert mechanism) must be flagged, not hidden."""
    s = _twin_snapshot_self(tmp_path)
    s._snapshot_twin_outputs(
        step=1,
        epoch=0,
        clean_norm=torch.zeros(1, 1, 4, 4, dtype=torch.complex64),
        corrupted_norm=torch.zeros(1, 1, 4, 4, dtype=torch.complex64),
        marker_residual=torch.zeros(1, 1, 4, 4, dtype=torch.complex64),
        marker_prior_norm=torch.zeros(1, 1, 4, 4, dtype=torch.complex64),
    )
    extra = _twin_extra(tmp_path, 1)
    assert extra["marker_mechanism_fired"] is False
    assert extra["marker_residual_abs_max"] == 0.0


def test_the_marker_reduction_is_never_paid_on_a_suppressed_step(tmp_path) -> None:
    """#1188: the twin's ``.item()`` must not touch the warm training loop.

    ``_snapshot_twin_outputs`` runs unconditionally from
    ``_compute_losses_impl``, i.e. on EVERY training step, and VF arms leave
    ``interval_steps`` at 0 -- so ``max_calls`` is the only gate. It lives
    inside ``save_debug_snapshot``, downstream of where the reduction used to be
    computed, and therefore suppressed the write while the sync was paid
    forever. Removing this method's own private budget for #706 is what exposed
    that: the private budget had been the only thing upstream of the sync.

    The probed steps below are budget-suppressed. Pre-fix this reads ``[1, 1, 1]``.
    """
    s = _twin_snapshot_self(tmp_path, max_calls=1)
    zeros = torch.zeros(1, 1, 4, 4, dtype=torch.complex64)
    # Burn the single allowance with a real tensor.
    s._snapshot_twin_outputs(
        step=0,
        epoch=0,
        clean_norm=zeros,
        corrupted_norm=zeros,
        marker_residual=zeros + 0.1,
        marker_prior_norm=zeros,
    )
    assert (tmp_path / "debug_snapshots" / "vf_twin_step_000000").is_dir()

    probes = []
    for step in (1, 2, 3):
        probe = _SyncCountingTensor(zeros + 0.1)
        probes.append(probe)
        s._snapshot_twin_outputs(
            step=step,
            epoch=0,
            clean_norm=zeros,
            corrupted_norm=zeros,
            marker_residual=probe,
            marker_prior_norm=zeros,
        )
    assert [p.syncs for p in probes] == [0, 0, 0]
    # ...and nothing was written for them either -- the budget still holds.
    assert not (tmp_path / "debug_snapshots" / "vf_twin_step_000003").exists()


def test_the_twin_docstring_does_not_claim_a_gate_it_does_not_own() -> None:
    """The docstring is half of what #1188 reports, so it is pinned too.

    It used to say the reduction was "gated to the first
    ``debug_snapshot_max_calls`` steps" -- naming a config key retired by
    RENAMES, and a STEP gate that never existed (``max_calls`` counts calls). A
    reader checking non-negotiable 9 against that sentence would have concluded
    the sync was already handled and moved on, which is what happened.
    """
    import inspect

    doc = inspect.getdoc(ConcreteVirtualFiducialStrategy._snapshot_twin_outputs) or ""
    assert "DEFERRED" in doc
    assert "debug_snapshot_max_calls" not in doc


# ── 2026-08-17: the DECLARATION half of the same contract (non-negotiable 14) ──
# The two tests above pin the EMISSION: `_snapshot_twin_outputs` writes `vf_twin`.
# What was never pinned is the class-level declaration that tells an artifact
# reader to go looking for it. VFS inherited `BaseTrainingStrategy`'s default
# `snapshot_prepared_is_model_input = True` while the comment right above its own
# twin call said the opposite -- the base captured the PRE-twin input and
# labelled it "the model input", so every artifact asserted the twin had not
# fired (pitfall #16, in machine-readable form contradicting the prose beside it).
#
# The assertion that matters is not either half alone; it is that they AGREE.


def test_declared_model_input_tag_matches_the_tag_actually_emitted(tmp_path) -> None:
    """Bind the declaration to the emission so they cannot drift apart.

    Asserting `snapshot_model_input_tag == "vf_twin"` on its own would only pin
    a string against itself. Comparing it to what `_snapshot_twin_outputs`
    really writes is what makes a future rename of either half fail loud.

    Read off DISK, not off a recorder. `save_debug_snapshot` names the directory
    `<tag>_step_<n>`, so the artifact itself carries the tag -- whereas a spy
    only ever shows the argument that was passed, and #1188 is precisely the
    case where the passed arguments and the written artifact disagreed. A
    dangling `snapshot_model_input_tag` is a pointer into the filesystem, so the
    filesystem is where it has to be resolved.
    """
    s = _twin_snapshot_self(tmp_path)
    step = 1
    s._snapshot_twin_outputs(
        step=step,
        epoch=0,
        clean_norm=_cplx(1, 1, 4, 4),
        corrupted_norm=_cplx(1, 1, 4, 4),
        marker_residual=_cplx(1, 1, 4, 4),
        marker_prior_norm=_cplx(1, 1, 4, 4),
    )

    assert ConcreteVirtualFiducialStrategy.snapshot_prepared_is_model_input is False

    tag = ConcreteVirtualFiducialStrategy.snapshot_model_input_tag
    written = sorted(p.name for p in (tmp_path / "debug_snapshots").iterdir() if p.is_dir())
    assert f"{tag}_step_{step:06d}" in written, (
        f"the declared tag {tag!r} names no artifact on disk; found {written} -- "
        "the carve-out points at a snapshot that was never written"
    )


def test_the_twin_snapshot_names_the_tensor_the_model_is_fed(tmp_path) -> None:
    """#1298: pointing at the right FILE is not pointing at the right TENSOR.

    `vf_twin` carries four tensors -- the clean twin, the corrupted input and
    the two marker halves. Only `twin_corrupted_input` reaches the network. The
    class docstring has always said so; this makes it a property of the
    artifact, so a future reordering of the dict cannot quietly relabel it.
    """
    import json

    s = _twin_snapshot_self(tmp_path)
    s._snapshot_twin_outputs(
        step=1,
        epoch=0,
        clean_norm=_cplx(1, 1, 4, 4),
        corrupted_norm=_cplx(1, 1, 4, 4),
        marker_residual=_cplx(1, 1, 4, 4),
        marker_prior_norm=_cplx(1, 1, 4, 4),
    )

    payload = json.loads(
        (tmp_path / "debug_snapshots" / "vf_twin_step_000001" / "snapshot.json").read_text()
    )
    verdict = payload["model_input_contract"]
    assert verdict["model_input_key"] == "twin_corrupted_input"
    # No generator on this bare instance, so the width half cannot be read --
    # and says so rather than reading as a pass.
    assert verdict["status"] == "unresolved"


def test_the_carve_out_is_declared_not_inherited_from_base() -> None:
    """The default is `True`; VFS must override it, not receive it.

    `ConcreteVirtualFiducialStrategy` extends `BaseTrainingStrategy` directly,
    so nothing upstream sets the carve-out for it — which is precisely how the
    false `True` survived. Reading the attribute off the class dict (rather than
    via inheritance) is what distinguishes "declared" from "happened to be".
    """
    for klass in ConcreteVirtualFiducialStrategy.__mro__:
        if "snapshot_prepared_is_model_input" in vars(klass):
            owner = klass
            break
    else:  # pragma: no cover - the attribute exists on BaseTrainingStrategy
        raise AssertionError("snapshot_prepared_is_model_input is unset entirely")

    assert owner is ConcreteVirtualFiducialStrategy, (
        f"the carve-out is inherited from {owner.__name__}, not declared here; "
        "an inherited value is what made every VF artifact claim the pre-twin "
        "input was the model input"
    )


# `test_twin_snapshot_skipped_past_call_cap` lived here and was red on `dev` (#826).
# Removed rather than repaired, because it asserted a budget this unit no longer owns
# and, as written, could not have passed against any implementation:
#
#   * #706 moved the allowance into `save_debug_snapshot`, keyed per (run_dir, TAG),
#     so `_snapshot_twin_outputs` deliberately keeps "No private budget here" (see its
#     docstring). The test stubbed `save_debug_snapshot` with a spy that has no budget
#     logic -- i.e. it replaced the enforcer, then asserted enforcement.
#   * It also conflated `step=9` with a CALL count: one call is call #1, under any cap.
#   * Its stub set `logging.debug_snapshot_max_calls`, retired by RENAMES to
#     `logging.snapshots.max_calls`, so it configured nothing regardless.
#
# No coverage is lost. The real seam is asserted where the budget actually lives:
# `tests/unit/infrastructure/training/test_debug_snapshot.py::test_max_calls_enforced`
# and `::test_each_tag_gets_its_own_budget` -- the latter being exactly #706's per-tag
# keying. The tests above still pin what this unit DOES own: that it publishes under
# the `vf_twin` tag with the `marker_mechanism_fired` / `marker_residual_abs_max`
# stamps.


# ── 2026-06-07: field self-consistency scoring (qMRI-claim test) ──────────────
# The DigitalTwinSimulator now exposes the B0 geometric-distortion field it
# applies; field-tracking models expose their shift estimate; the VF strategy
# scores them so the headline qMRI claim is actually tested (not image-recon
# only). These pin the seam end-to-end without the heavyweight training stack.

from spectramr.infrastructure.physics.digital_twin_simulator import (  # noqa: E402
    simulate_b0_geometric_distortion,
)


def test_twin_b0_distortion_exposes_field_when_requested():
    img = torch.randn(2, 1, 64, 64, dtype=torch.complex64)
    # default return is unchanged (single complex tensor) — no caller breakage.
    assert torch.is_complex(simulate_b0_geometric_distortion(img, (64, 64)))
    out = simulate_b0_geometric_distortion(
        img, (64, 64), b0_strength=0.3, return_field=True, sequence="epi"
    )
    assert isinstance(out, tuple) and len(out) == 3
    dist, b0_map, pe_shift = out
    assert torch.is_complex(dist) and dist.shape == img.shape
    assert b0_map.shape == (2, 64, 64) and pe_shift.shape == (2, 64, 64)
    # EPI: the PE shift is 8x the FE shift, so the tracker has a displacement to recover.
    assert pe_shift.abs().mean() > 0


def _vf_with_field(twin_pe_shift, est):
    s = _VFS.__new__(_VFS)
    s.simulator = SimpleNamespace(last_pe_shift_field=twin_pe_shift)
    s.env = SimpleNamespace(generator=SimpleNamespace(last_shift_estimate=est))
    return s


def test_score_field_zero_when_estimate_matches_twin():
    _, _, pe = simulate_b0_geometric_distortion(
        torch.randn(3, 1, 64, 64, dtype=torch.complex64), (64, 64), return_field=True,
        sequence="epi",  # the tracker scores the EPI phase-encode shift
    )
    twin_char = pe.abs().flatten(1).mean(dim=1)
    est = twin_char / (64 / 2.0)  # invert px→normalised so the model "matches"
    m = _VFS._score_field(_vf_with_field(pe, est))
    assert m["val_field_shift_mae"] < 1e-3
    assert abs(m["val_field_shift_bias"]) < 1e-3
    assert m["val_twin_pe_shift_px"] > 0


def test_score_field_penalises_wrong_estimate():
    _, _, pe = simulate_b0_geometric_distortion(
        torch.randn(3, 1, 64, 64, dtype=torch.complex64), (64, 64), return_field=True,
        sequence="epi",  # the tracker scores the EPI phase-encode shift
    )
    twin_char = pe.abs().flatten(1).mean(dim=1)
    bad = (twin_char * 5.0) / (64 / 2.0)  # 5× too large
    m = _VFS._score_field(_vf_with_field(pe, bad))
    assert m["val_field_shift_mae"] > m["val_twin_pe_shift_px"]  # error exceeds the truth scale


def test_score_field_noop_for_non_field_arms():
    # No twin field (non-EPI arm) → empty, so val_psnr-only arms are unaffected.
    assert _VFS._score_field(_vf_with_field(None, torch.tensor([0.1]))) == {}
    assert _VFS._score_field(_vf_with_field(torch.zeros(1, 1, 8, 8), None)) == {}


# ── real-reference seam (B0 from real data, not a random sim) ─────────────────


def test_score_field_flags_real_vs_self_consistency():
    """val_field_reference_real distinguishes real-reference grading (the twin
    applied a REAL B0 map) from self-consistency on a synthetic field."""
    pe = torch.full((4, 8, 8), 3.0)
    est = torch.tensor([0.05, 0.05, 0.05, 0.05])
    real = _VFS._score_field(_vf_with_field(pe, est), real_reference=True)
    assert real["val_field_reference_real"] == 1.0
    assert "val_field_shift_mae" in real
    selfc = _VFS._score_field(_vf_with_field(pe, est), real_reference=False)
    assert selfc["val_field_reference_real"] == 0.0


def test_slice_field_reduces_5d_volume_and_passes_none():
    assert _VFS._slice_field(None) is None
    assert tuple(_VFS._slice_field(torch.zeros(1, 1, 4, 4, 6)).shape) == (1, 1, 4, 4)
    assert tuple(_VFS._slice_field(torch.zeros(1, 1, 4, 4)).shape) == (1, 1, 4, 4)


# ── phase-path arms: grade a field estimate vs a real B0's structure ──────────


def _vf_phase(b0_field, field_est):
    s = _VFS.__new__(_VFS)
    s.simulator = SimpleNamespace(last_pe_shift_field=None, last_b0_field=b0_field)
    s.env = SimpleNamespace(
        generator=SimpleNamespace(last_shift_estimate=None, last_field_estimate=field_est)
    )
    return s


def test_score_field_structure_grades_real_b0_structure():
    ref = torch.randn(2, 1, 16, 16)
    out = _VFS._score_field(_vf_phase(ref, ref.clone() * 3.0), real_reference=True)
    assert out["val_field_reference_real"] == 1.0
    assert out["val_field_b0_corr"] > 0.99  # identical structure (scale-invariant)
    assert out["val_field_b0_nrmse"] < 1e-3  # scale-fit removes the 3x


def test_score_field_structure_noop_without_real_reference_or_estimate():
    ref = torch.randn(1, 1, 8, 8)
    assert _VFS._score_field(_vf_phase(ref, ref), real_reference=False) == {}
    assert _VFS._score_field(_vf_phase(ref, None), real_reference=True) == {}


def test_phase_models_expose_last_field_estimate():
    from spectramr.models.generators.vf_field_generators import (
        GraphCutUnwrapGenerator,
        PhaseTrackingLSTM,
    )

    x = torch.randn(1, 2, 64, 64)
    for cls in (PhaseTrackingLSTM, GraphCutUnwrapGenerator):
        m = cls(in_channels=2, out_channels=2)
        m(x)
        assert getattr(m, "last_field_estimate", None) is not None
        assert m.last_field_estimate.shape[-2:] == (64, 64)


def test_the_twin_snapshot_is_emitted_unconditionally() -> None:
    """VF satisfies non-negotiable 14 through a SIDE CHANNEL, so this matters.

    Every other carve-out strategy calls ``_declare_model_input`` and the wrapper
    emits. VF instead relies on ``_snapshot_twin_outputs`` already writing the
    ``vf_twin`` tag, which sets the satisfaction flag inside
    ``save_debug_snapshot``. That only holds while the call is UNCONDITIONAL: put
    it behind ``if self.config...:`` and, on an arm with snapshots enabled and
    that branch false, the wrapper raises on a perfectly healthy run.

    The sibling test above calls ``_snapshot_twin_outputs`` directly, so it pins
    the tag but would stay green through exactly that regression. Asserting the
    call is a direct statement of the method body is what closes it — a call
    nested in any ``if``/``try``/``for`` is not a child of ``fn.body`` (AST, not
    a call-site regex, because indentation-matching is what gets this wrong).
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(ConcreteVirtualFiducialStrategy._compute_losses_impl))
    fn = ast.parse(src).body[0]
    assert isinstance(fn, ast.FunctionDef)

    def _is_twin_call(node: ast.stmt) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "_snapshot_twin_outputs"
        )

    assert any(_is_twin_call(stmt) for stmt in fn.body), (
        "_snapshot_twin_outputs is no longer a direct statement of "
        "_compute_losses_impl. VF's model-input contract is satisfied by that "
        "call alone, so guarding it makes the wrapper raise mid-run. Either "
        "restore the unconditional call or switch VF to _declare_model_input."
    )


# ── 2026-09-03 (VF review): out-of-distribution acceleration readout ─────────
# ``physics.digital_twin.ood_acceleration_range`` used to be the unread
# ``undersampling.out_of_distribution_range`` (58 arms, dropped-key baseline).
# ``_score_at_current_twin`` is the one corrupt-reconstruct-score pass; the
# readout holds the REAL twin at each rung and re-keys what the pass returns.


def _ood_vfs(rng):
    """A bare VFS on a real twin (4x in distribution) with an identity generator."""
    from spectramr.infrastructure.physics.digital_twin_simulator import DigitalTwinSimulator

    class _Probe(ConcreteVirtualFiducialStrategy):
        validation_metrics_computer = None  # the base property needs a validation block

    s = _Probe.__new__(_Probe)
    s.env = SimpleNamespace(generator=_PlainModel())
    s._conditioning_context = None
    s._ensure_conditioner = lambda n: None
    s.device = torch.device("cpu")
    s.simulator = DigitalTwinSimulator(
        im_size=(32, 32),
        enable_motion=False,
        snr_range=(100.0, 100.0),
        enable_undersampling=True,
        acceleration=4.0,
    )
    s.config = SimpleNamespace(
        model=SimpleNamespace(conditioning=ConditioningConfig(), target_domain="image"),
        physics=SimpleNamespace(
            digital_twin=SimpleNamespace(ood_acceleration_range=rng, enable_undersampling=True)
        ),
    )
    return s


def test_ood_readout_scores_every_rung_on_the_real_twin_and_restores_it() -> None:
    from spectramr.infrastructure.training.strategies.ood_acceleration_readout import (
        ood_acceleration_readout,
        ood_accelerations,
    )

    strat = _ood_vfs([16.0, 32.0])
    target = torch.complex(torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32))
    with torch.no_grad():
        scored = strat._score_at_current_twin(target, None, cache_visuals=True)
        out = ood_acceleration_readout(
            strat.simulator,
            ood_accelerations(strat.config),
            lambda: strat._score_at_current_twin(target, None, cache_visuals=False),
        )
    assert set(scored) == {"val_psnr"} and torch.isfinite(torch.tensor(scored["val_psnr"]))
    assert strat._last_visual_pred is not None  # the in-distribution pass caches visuals
    assert set(out) == {"val_ood_16x_psnr", "val_ood_32x_psnr", "val_ood_accelerations"}
    assert out["val_ood_accelerations"] == 2.0
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values())
    assert strat.simulator.acceleration == 4.0 and strat.simulator.enable_undersampling is True


def test_no_declared_range_gives_a_zero_count_and_no_extra_pass() -> None:
    from spectramr.infrastructure.training.strategies.ood_acceleration_readout import (
        ood_acceleration_readout,
        ood_accelerations,
    )

    strat = _ood_vfs(None)
    out = ood_acceleration_readout(
        strat.simulator, ood_accelerations(strat.config), lambda: pytest.fail("no rung, no pass")
    )
    assert out == {"val_ood_accelerations": 0.0}


def test_validation_step_scores_the_field_before_the_ood_rungs() -> None:
    """The field self-consistency reads the twin's LAST fields: it must run before the rungs."""
    import inspect

    src = inspect.getsource(ConcreteVirtualFiducialStrategy.validation_step)
    assert src.index("self._score_field(") < src.index("ood_acceleration_readout(")


def test_vf_strategy_reads_the_range_and_does_not_claim_the_undersampling_block() -> None:
    """The twin owns the acceleration; the top-level block reaches nothing here."""
    assert ConcreteVirtualFiducialStrategy.reads_ood_acceleration_range is True
    assert ConcreteVirtualFiducialStrategy.applies_undersampling is False
    assert "applies_undersampling" not in ConcreteVirtualFiducialStrategy.__dict__


def test_severity_vector_reports_the_rate_the_twin_is_held_at() -> None:
    """The conditioning context is rebuilt per pass, so a severity-conditioned model is
    told the OOD rate inside ``at_acceleration`` and the training rate outside it."""
    strat = _ood_vfs([16.0])
    device = torch.device("cpu")
    outside = _VF._severity_vector(strat.simulator, 1, device)[0, -1].item()
    with strat.simulator.at_acceleration(16.0):
        inside = _VF._severity_vector(strat.simulator, 1, device)[0, -1].item()
    assert abs(outside - (4.0 - 1.0) / 15.0) < 1e-6
    assert abs(inside - 1.0) < 1e-6


# ── #867: phase metrics at VF validation see the complex pair ─────────────────
# ``_score_at_current_twin`` scored the metric set on magnitudes, so a declared
# ``phase_mse`` (exp_vf_04) raised PhaseBlindInputError and the strategy's broad
# ``except`` reduced the step to ``val_psnr`` alone.


def _vfs_with_phase_metrics():
    from spectramr.core.metrics.computer import ValidationMetricsComputer
    from spectramr.core.metrics.types import ValidationMetricsConfig

    strat = _ood_vfs([])
    strat.validation_metrics_computer = ValidationMetricsComputer(
        ValidationMetricsConfig.from_dict({"metrics": ["psnr", "phase_mse"]}), device="cpu"
    )
    return strat


def test_phase_metrics_are_scored_on_the_complex_pair() -> None:
    strat = _vfs_with_phase_metrics()
    target = torch.complex(torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32))
    with torch.no_grad():
        scored = strat._score_at_current_twin(target, None, cache_visuals=False)
    assert {"val_psnr", "psnr", "phase_mse"} <= set(scored)
    assert torch.isfinite(torch.tensor(scored["phase_mse"]))


def test_a_phase_blind_metric_set_raises_instead_of_degrading(monkeypatch) -> None:
    """Planted: the pre-fix magnitude-only call must fail loudly, not leave val_psnr alone."""
    from spectramr.core.metrics.complex_input import PhaseBlindInputError
    from spectramr.infrastructure.training.strategies import virtual_fiducial_strategy as vfs
    from spectramr.infrastructure.training.utils.metric_views import MetricViews

    monkeypatch.setattr(vfs, "MetricViews", lambda p, t, *_: MetricViews(p, t))
    strat = _vfs_with_phase_metrics()
    target = torch.complex(torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32))
    with torch.no_grad(), pytest.raises(PhaseBlindInputError):
        strat._score_at_current_twin(target, None, cache_visuals=False)


def test_the_conditioner_is_declared_exactly_when_it_is_built():
    """The audit and the builder share one predicate (#2629): declared as a bare group only
    for a config that builds the conditioner, so VF without conditioning is not refused."""
    on = SimpleNamespace(
        model=SimpleNamespace(
            conditioning=ConditioningConfig(enabled=True, sources=["severity_vec"], embed_dim=8)
        )
    )
    off = SimpleNamespace(model=SimpleNamespace(conditioning=ConditioningConfig()))
    assert ("_conditioner", "param_group") in _VF.owned_module_declarations(on)
    assert _VF.owned_module_declarations(off) == ()
