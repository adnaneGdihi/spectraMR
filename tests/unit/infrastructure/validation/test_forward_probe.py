"""Unit tests for the Tier-2 synthetic forward probe.

The probe is intentionally tolerant: missing torch, missing model
registry entries, and missing config attributes all surface as
structured :class:`ProbeResult` records, never as raised exceptions
that would crash the audit CLI.
"""

from __future__ import annotations

import json
import types
from typing import Any, ClassVar

import pytest

from spectramr.models.registry import register_model as _register_model
from tests.utils.config_block_stub import block_stub
from tests.utils.data_config_stub import DataConfigStub

torch = pytest.importorskip("torch")

from spectramr.infrastructure.validation.forward_probe import (  # noqa: E402
    ProbeResult,
    _loss_gradient_verdict,
    synthetic_forward_probe,
)

# ── Configured-loss gradient-fires verdict (dead_loss audit gate) ───────


def test_loss_gradient_verdict_zero_total_is_dead_loss() -> None:
    """A configured g_total_loss of exactly 0.0 → dead_loss (no gradient)."""
    total = torch.tensor(0.0, requires_grad=True)
    r = _loss_gradient_verdict(total, "c_mno_operator", "cpu", 0.0)
    assert r is not None and r.passed is False and r.category == "dead_loss"


def test_loss_gradient_verdict_none_total_is_dead_loss() -> None:
    """A None total (no loss component aggregated) → dead_loss."""
    r = _loss_gradient_verdict(None, "x", "cpu", 0.0)
    assert r is not None and r.category == "dead_loss"


def test_loss_gradient_verdict_no_grad_total_is_dead_loss() -> None:
    """A finite, nonzero total that does not require grad → dead_loss."""
    total = torch.tensor(0.5, requires_grad=False)
    r = _loss_gradient_verdict(total, "x", "cpu", 0.0)
    assert r is not None and r.category == "dead_loss"


def test_loss_gradient_verdict_healthy_total_passes() -> None:
    """A finite, nonzero, grad-carrying total → no verdict (continue)."""
    total = torch.tensor(1.5, requires_grad=True)
    assert _loss_gradient_verdict(total, "x", "cpu", 0.0) is None


def test_loss_gradient_verdict_nan_total_deferred() -> None:
    """A NaN total is left to the existing nan_in_loss path (verdict None)."""
    total = torch.tensor(float("nan"), requires_grad=True)
    assert _loss_gradient_verdict(total, "x", "cpu", 0.0) is None

# ── ProbeResult JSON-serialisation contract ────────────────────────────


def test_probe_result_to_dict_contains_required_keys() -> None:
    r = ProbeResult(passed=True, category="forward_pass_shape", message="ok")
    d = r.to_dict()
    for key in (
        "passed", "category", "message", "severity", "yaml_keys",
        "fix_hint", "device", "elapsed_seconds", "traceback",
    ):
        assert key in d


def test_probe_result_to_json_round_trips() -> None:
    r = ProbeResult(passed=False, category="oom", message="cuda oom",
                    yaml_keys=["data.batch_size"], fix_hint="lower batch size")
    parsed = json.loads(r.to_json())
    assert parsed["passed"] is False
    assert parsed["category"] == "oom"
    assert parsed["yaml_keys"] == ["data.batch_size"]


# ── Forward-probe behaviour ────────────────────────────────────────────


def _cfg(
    model_type: str = "test_probe_model",
    in_channels: int = 1,
    out_channels: int = 1,
    patch_size: list[int] | None = None,
    batch_size: int = 1,
    model_kwargs: dict | None = None,
    spatial_dims: int | None = None,
) -> Any:
    model_ns = types.SimpleNamespace(
        model_type=model_type,
        in_channels=in_channels,
        out_channels=out_channels,
        model_kwargs=model_kwargs or {},
    )
    if spatial_dims is not None:
        model_ns.spatial_dims = spatial_dims
    return types.SimpleNamespace(
        model=model_ns,
        # `data.sampling.patch_size` / `data.loader.batch_size` since the block
        # decomposition -- forward_probe walks the canonical paths, so a flat
        # stand-in raises `no attribute 'sampling'` before the probe ever runs.
        data=DataConfigStub(
            patch_size=patch_size or [16, 16],
            batch_size=batch_size,
        ),
    )


class _GoodModel(torch.nn.Module):
    """Identity-style 2D conv that preserves shape."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.conv(x)


class _ShapeBreakingModel(torch.nn.Module):
    """Returns wrong spatial size — exercises the shape-mismatch path."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _RaisingForwardModel(torch.nn.Module):
    """Forward unconditionally raises — exercises the runtime-error path."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (65536x2 and 256x8)")


class _ContrastConditionedModel(torch.nn.Module):
    """Field+contrast-conditioned net that RAISES when contrast_id is absent.

    Mirrors FieldVelocityUNet(use_contrast_conditioning=True): ``field_strength``
    is required (no default), ``contrast_id`` has a default (None) but the model
    fails loud on None when conditioning is on (#15). The probe must synthesise a
    valid contrast_id despite its default, or the mechanism-fires probe can never
    exercise a contrast-conditioned arm.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        use_contrast_conditioning: bool = False,
        num_contrasts: int = 3,
        **kwargs: Any,
    ):
        super().__init__()
        self.use_contrast_conditioning = use_contrast_conditioning
        self.num_contrasts = num_contrasts
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        contrast_id: Any = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.use_contrast_conditioning:
            if contrast_id is None:
                raise ValueError("requires a per-sample 'contrast_id' batch key")
            if (
                int(contrast_id.min()) < 0
                or int(contrast_id.max()) >= self.num_contrasts
            ):
                raise ValueError("contrast_id out of range")
        return self.conv(x)


class _StyleConditionedModel(torch.nn.Module):
    """AdaIN generator requiring a style vector ``s`` (StarGAN v2 contract).

    Mirrors :class:`StarGANv2Generator.forward(x, s)`: ``s`` is a REQUIRED
    positional arg (no default) that the training strategy computes from a
    mapping network / style encoder the probe does not run. The probe must
    synthesise a ``[B, style_dim]`` code — reading ``style_dim`` off the model
    (declared as ``self.style_dim``), not a hardcoded constant. This model
    RAISES on a wrong-width style so the test proves the read is honoured.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        style_dim: int = 64,
        **kwargs: Any,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.to_scale = torch.nn.Linear(style_dim, out_channels)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if s.ndim != 2 or s.shape[1] != self.style_dim:
            raise ValueError(
                f"style vector must be [B, {self.style_dim}], got {tuple(s.shape)}"
            )
        h = self.conv(x)
        scale = self.to_scale(s).unsqueeze(-1).unsqueeze(-1)
        return h * (1.0 + scale)


@pytest.fixture
def _patch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the test models against the lazy registry import.

    The probe imports ``get_model_class`` lazily inside its body
    (``from spectramr.models.registry import get_model_class``), so we patch
    the function directly on the registry module.
    """
    from spectramr.models import registry as registry_mod

    fake = {
        "good": _GoodModel,
        "shape_break": _ShapeBreakingModel,
        "raising": _RaisingForwardModel,
        "contrast_cond": _ContrastConditionedModel,
        "style_cond": _StyleConditionedModel,
    }

    def _fake_get_model_class(name: str) -> Any:
        if name not in fake:
            raise KeyError(f"{name!r} not registered")
        return fake[name]

    monkeypatch.setattr(registry_mod, "get_model_class", _fake_get_model_class)


def test_probe_passes_on_well_formed_model(_patch_registry: None) -> None:
    cfg = _cfg(model_type="good")
    result = synthetic_forward_probe(cfg, device="cpu", backward=True)
    assert result.passed
    assert result.category == "forward_pass_shape"
    assert result.elapsed_seconds >= 0.0


def test_probe_fails_on_unregistered_model(_patch_registry: None) -> None:
    cfg = _cfg(model_type="never_registered")
    result = synthetic_forward_probe(cfg, device="cpu")
    assert not result.passed
    assert result.category == "instantiation"
    assert "never_registered" in result.message


def test_probe_fails_on_shape_mismatch(_patch_registry: None) -> None:
    cfg = _cfg(model_type="shape_break", patch_size=[16, 16])
    result = synthetic_forward_probe(cfg, device="cpu")
    assert not result.passed
    assert result.category == "forward_pass_shape"
    assert "shape" in result.message.lower()
    assert "data.patch_size" in result.yaml_keys or "model.out_channels" in result.yaml_keys


def test_probe_classifies_runtime_error_as_forward_pass_shape(_patch_registry: None) -> None:
    cfg = _cfg(model_type="raising")
    result = synthetic_forward_probe(cfg, device="cpu")
    assert not result.passed
    assert result.category == "forward_pass_shape"
    assert "mat1" in result.message  # the exception bubbled into the message


def test_probe_skips_when_model_section_incomplete() -> None:
    bad = types.SimpleNamespace(model=types.SimpleNamespace(), data=types.SimpleNamespace())
    result = synthetic_forward_probe(bad, device="cpu")
    assert not result.passed
    assert result.category == "instantiation"


def test_probe_synthesizes_contrast_id_for_conditioned_model(_patch_registry: None) -> None:
    # The mechanism-fires probe must feed a valid per-sample contrast_id to a
    # contrast-conditioned model, even though contrast_id has a default (None):
    # the model raises loud on None (#15), so a missing synthesis would fail the
    # probe on EVERY contrast-conditioned arm.
    cfg = _cfg(
        model_type="contrast_cond",
        model_kwargs={"use_contrast_conditioning": True, "num_contrasts": 3},
    )
    result = synthetic_forward_probe(cfg, device="cpu", backward=True)
    assert result.passed, result.message
    assert result.category == "forward_pass_shape"


def test_probe_contrast_blind_model_unaffected(_patch_registry: None) -> None:
    # A contrast-BLIND instance still probes cleanly (contrast_id is synthesised
    # but ignored — the model never requires it).
    cfg = _cfg(
        model_type="contrast_cond",
        model_kwargs={"use_contrast_conditioning": False},
    )
    result = synthetic_forward_probe(cfg, device="cpu", backward=True)
    assert result.passed, result.message


def test_probe_synthesizes_style_vector_for_stargan_like_model(
    _patch_registry: None,
) -> None:
    # StarGANv2Generator.forward(x, s) needs a strategy-computed style vector the
    # probe never runs the strategy to produce. The probe must synthesise a
    # [B, style_dim] code from the REQUIRED positional ``s`` — reading style_dim
    # off the model (here 32, not the default 64) — so the generator is actually
    # probed instead of TypeError-failing Tier-2 on the missing arg.
    cfg = _cfg(model_type="style_cond", model_kwargs={"style_dim": 32})
    result = synthetic_forward_probe(cfg, device="cpu", backward=True)
    assert result.passed, result.message
    assert result.category == "forward_pass_shape"


# ── Warnings-and-fallbacks probe extensions ────────────────────────────


class _IdentityModel(torch.nn.Module):
    """Returns input verbatim — exercises the identity-collapse path."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _NaNModel(torch.nn.Module):
    """Returns NaN — exercises the nan_in_forward path."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * float("nan")


class _ExplodingModel(torch.nn.Module):
    """Huge initialisation -> gradient explosion on first backward.

    A single Conv2d isn't enough to trigger the probe's 1e6 grad-norm
    threshold under an L1 loss — the L1 output-gradient caps at
    ``sign(y-t)/N`` so the chain rule needs additional amplification.
    Squaring the output adds a ``2·y`` factor and an 8× weight scaling
    on a Shepp-Logan phantom (sparse, mean magnitude ~0.1) gives a
    grad-norm of several megabytes, well past the explosion threshold.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.mul_(1e8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y * y


@pytest.fixture
def _patch_registry_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from spectramr.models import registry as registry_mod

    fake = {
        "identity":  _IdentityModel,
        "nan":       _NaNModel,
        "exploding": _ExplodingModel,
    }
    def _fake_get_model_class(name: str) -> Any:
        return fake[name]
    monkeypatch.setattr(registry_mod, "get_model_class", _fake_get_model_class)


def test_probe_flags_identity_collapse(_patch_registry_extra: None) -> None:
    cfg = _cfg(model_type="identity")
    result = synthetic_forward_probe(cfg, device="cpu")
    assert not result.passed
    assert result.category == "identity_collapse"
    assert result.severity == "warning"
    assert "identity" in result.message.lower()


def test_probe_flags_nan_in_forward(_patch_registry_extra: None) -> None:
    cfg = _cfg(model_type="nan")
    result = synthetic_forward_probe(cfg, device="cpu")
    assert not result.passed
    assert result.category == "nan_in_forward"
    assert result.severity == "error"


def test_probe_flags_gradient_explosion(_patch_registry_extra: None) -> None:
    cfg = _cfg(model_type="exploding", patch_size=[16, 16])
    result = synthetic_forward_probe(cfg, device="cpu", backward=True)
    assert not result.passed
    assert result.category == "gradient_explosion"
    assert result.severity == "warning"
    assert "1e+06" in result.message or "1.0e+06" in result.message or "1e6" in result.message.lower()


# ── Phantom input vs. white-noise input ─────────────────────────────────


class _CaptureInputModel(torch.nn.Module):
    """Stash the input tensor on the class so the test can inspect it."""

    last_input: torch.Tensor | None = None

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Capture the FIRST input only: the input-invariance probe runs extra
        # forwards (a re-run + a different input) after the primary one, and the
        # phantom/noise tests assert on the *primary* synthetic input.
        if type(self).last_input is None:
            type(self).last_input = x.detach().clone()
        return self.conv(x)


@pytest.fixture
def _patch_registry_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    from spectramr.models import registry as registry_mod
    def _fake_get_model_class(name: str) -> Any:
        assert name == "capture"
        return _CaptureInputModel
    monkeypatch.setattr(registry_mod, "get_model_class", _fake_get_model_class)


def test_probe_uses_phantom_by_default(_patch_registry_capture: None) -> None:
    """Default input must look like a phantom, not noise: bounded [0,1]
    and far fewer unique values than a 1024-element noise tensor."""
    _CaptureInputModel.last_input = None
    cfg = _cfg(model_type="capture", patch_size=[32, 32])
    result = synthetic_forward_probe(cfg, device="cpu", backward=False)
    assert result.passed
    x = _CaptureInputModel.last_input
    assert x is not None
    assert x.min().item() >= 0.0
    assert x.max().item() <= 1.0 + 1e-6
    n_unique = torch.unique(x).numel()
    assert n_unique < 50, (
        f"Default probe input should be a Shepp-Logan phantom (~10 levels), "
        f"got {n_unique} unique values — looks like noise."
    )


def test_probe_uses_noise_when_use_phantom_false(_patch_registry_capture: None) -> None:
    """Opt-in noise input must be unbounded gaussian."""
    _CaptureInputModel.last_input = None
    cfg = _cfg(model_type="capture", patch_size=[32, 32])
    result = synthetic_forward_probe(cfg, device="cpu", backward=False, use_phantom=False)
    assert result.passed
    x = _CaptureInputModel.last_input
    assert x is not None
    # Random gaussian: at this size, > 100 unique values is virtually certain.
    assert torch.unique(x).numel() > 100


class _SpatialDims3DModel(torch.nn.Module):
    """Conv3d-only model. Crashes with conv2d if instantiated as 2D."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_dims: int = 2,
        **kwargs: Any,
    ):
        super().__init__()
        if spatial_dims == 3:
            self.conv = torch.nn.Conv3d(in_channels, out_channels, 3, padding=1)
        else:
            self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.spatial_dims = spatial_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _DiffusionLikeModel(torch.nn.Module):
    """Model whose forward requires a ``timesteps`` kwarg. Mirrors the
    ULF stage-2 LatentDiffusionGenerator signature contract."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # The ``timesteps`` arg must be supplied; otherwise plain ``model(x)``
        # raises TypeError. We fold it into the bias so the test can verify
        # the probe actually fed something nonzero.
        return self.conv(x) + timesteps.float().mean()


@pytest.fixture
def _patch_registry_signatures(monkeypatch: pytest.MonkeyPatch) -> None:
    from spectramr.models import registry as registry_mod

    fake = {
        "spatial3d": _SpatialDims3DModel,
        "diffusion_like": _DiffusionLikeModel,
    }

    def _fake_get_model_class(name: str) -> Any:
        return fake[name]

    monkeypatch.setattr(registry_mod, "get_model_class", _fake_get_model_class)


def test_probe_passes_top_level_spatial_dims_to_constructor(
    _patch_registry_signatures: None,
) -> None:
    """Regression: ``model.spatial_dims`` is a top-level YAML field but
    the prior probe only forwarded ``model.model_kwargs.*`` to the
    constructor. The ULF stage-1 cluster failure was a 3D model
    silently defaulting to ``spatial_dims=2``, then crashing conv2d on
    a 5D patch tensor. The probe must thread the top-level field
    through so 3D models actually instantiate as 3D.
    """
    cfg = _cfg(
        model_type="spatial3d",
        patch_size=[16, 16, 8],  # 3D patch
        spatial_dims=3,
    )
    result = synthetic_forward_probe(cfg, device="cpu")
    assert result.passed, (
        f"3D probe should pass; got {result.category}: {result.message}"
    )


def test_probe_supplies_timesteps_for_diffusion_models(
    _patch_registry_signatures: None,
) -> None:
    """Regression: diffusion / latent-diffusion forward signatures
    require a ``timesteps`` kwarg. The prior probe called ``model(x)``
    flat and TypeErrored on the ULF stage-2 LDM. The probe now
    introspects ``forward.__signature__`` and supplies a synthetic
    timestep tensor for any required parameter named timestep / t / ts.
    """
    cfg = _cfg(model_type="diffusion_like", patch_size=[16, 16])
    result = synthetic_forward_probe(cfg, device="cpu")
    assert result.passed, (
        f"diffusion-like probe should pass; got "
        f"{result.category}: {result.message}"
    )


def test_probe_save_images_writes_pngs(
    tmp_path: pytest.TempPathFactory, _patch_registry_capture: None
) -> None:
    pytest.importorskip("matplotlib")
    cfg = _cfg(model_type="capture", patch_size=[16, 16])
    out_dir = str(tmp_path)
    result = synthetic_forward_probe(
        cfg, device="cpu", backward=False,
        save_images_dir=out_dir, arm_name="arm42",
    )
    assert result.passed
    import os
    pngs = sorted(p for p in os.listdir(out_dir) if p.endswith(".png"))
    # At least input + output (target may be skipped if missing).
    assert any("arm42_input" in p for p in pngs)
    assert any("arm42_output" in p for p in pngs)


# ── Input-invariance (measurement-independent / facade) guard ──────────
# G1 of the 2026-06 scientific-validation prevention layer: a model whose
# output ignores its input is a facade (DC-blob class) that smoke never catches.

from spectramr.infrastructure.validation.forward_probe import (  # noqa: E402
    _input_invariance_stats,
    _is_input_invariant,
)


class TestInputInvarianceLogic:
    """Pure decision logic — no model build needed (the runtime probe wires it in)."""

    def test_constant_output_is_invariant(self) -> None:
        y = torch.ones(1, 1, 8, 8) * 0.5
        d_same, d_diff = _input_invariance_stats(y, y.clone(), y.clone())
        assert d_same == pytest.approx(0.0) and d_diff == pytest.approx(0.0)
        assert _is_input_invariant(d_same, d_diff) is True

    def test_input_sensitive_output_passes(self) -> None:
        y = torch.randn(1, 1, 8, 8)
        y_same = y.clone()  # deterministic re-run
        y_diff = torch.randn(1, 1, 8, 8)  # different input → different output
        d_same, d_diff = _input_invariance_stats(y, y_same, y_diff)
        assert _is_input_invariant(d_same, d_diff) is False

    def test_stochastic_facade_flagged_by_ratio(self) -> None:
        # Output varies (dropout-like noise floor) but NO MORE with a changed input.
        assert _is_input_invariant(d_same=0.05, d_diff=0.05) is True

    def test_stochastic_sensitive_passes(self) -> None:
        # Changed input moves the output far beyond the noise floor.
        assert _is_input_invariant(d_same=0.05, d_diff=0.6) is False


class _ConstantModel(torch.nn.Module):
    """Ignores its input — returns a constant (measurement-independent facade)."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x) * self.bias


def _patch_one(monkeypatch: pytest.MonkeyPatch, name: str, cls: Any) -> None:
    from spectramr.models import registry as registry_mod

    monkeypatch.setattr(
        registry_mod,
        "get_model_class",
        lambda n: cls if n == name else (_ for _ in ()).throw(KeyError(n)),
    )


def test_probe_flags_input_invariant_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_one(monkeypatch, "facade", _ConstantModel)
    result = synthetic_forward_probe(_cfg(model_type="facade"), device="cpu", backward=True)
    assert not result.passed
    assert result.category == "input_invariant"
    assert result.severity == "warning"


def test_probe_input_sensitive_model_not_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_one(monkeypatch, "good", _GoodModel)
    result = synthetic_forward_probe(_cfg(model_type="good"), device="cpu", backward=True)
    # A real conv responds to its input → must pass the whole probe.
    assert result.passed
    assert result.category != "input_invariant"


# ── synthetic_forward_probe_skip type-robustness + mid-schedule t ──────
# (2026-06 infrastructure audit)


_CAPTURED_TIMESTEP: dict[str, Any] = {}


class _BoolSkipIdentityModel(torch.nn.Module):
    """Identity model that opts out of the probe via a *bool* skip flag.

    ``synthetic_forward_probe_skip`` is documented as ``set[str]`` but at
    least one model (``bloch_manifold_projector``) declares it ``bool``.
    With the old ``"identity_collapse" not in <bool>`` membership test this
    raised an uncaught ``TypeError`` that crashed the whole audit.
    """

    synthetic_forward_probe_skip = True

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        # A trainable scalar initialised to 1.0 keeps the output ≈ identity
        # (so the identity-collapse check WOULD fire if not skipped) while
        # giving the probe's backward pass a parameter to differentiate.
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _TimestepCaptureModel(torch.nn.Module):
    """Requires a ``timestep`` arg and records the value the probe passes."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        _CAPTURED_TIMESTEP["t"] = timestep
        return self.conv(x)


class _OptionalTimestepModel(torch.nn.Module):
    """``timesteps`` carries a default and the model still needs a real value.

    The shape of every real diffusion generator's forward — and the shape the
    probe's "only fill REQUIRED args" filter silently skipped, so the model was
    always probed with ``timesteps=None``. ``KSpaceColdDiffusionGenerator``
    responds to that by warning and DEGRADING to t=0, the fully-denoised
    boundary at which a cold-diffusion net legitimately approaches identity, so
    the probe was grading the one timestep where the time embedding cannot be
    wrong.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        _CAPTURED_TIMESTEP["t"] = timesteps
        return self.conv(x)


class _KwargsContractModel(torch.nn.Module):
    """Requires a ``**kwargs``-only kwarg and DECLARES it to the probe.

    ``inspect.signature`` cannot enumerate names behind ``VAR_KEYWORD``, so a
    signature-driven probe can never discover this contract. The model announces
    it via ``synthetic_forward_probe_kwargs``.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def synthetic_forward_probe_kwargs(self, x: torch.Tensor) -> dict[str, Any]:
        return {"measured": torch.zeros_like(x), "timesteps": torch.full((1,), 999)}

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if kwargs.get("measured") is None:
            raise ValueError("declared mechanism received no `measured`")
        _CAPTURED_TIMESTEP["hook_t"] = kwargs.get("timesteps")
        return self.conv(x)


class _KwargsContractModelNoHook(_KwargsContractModel):
    """Same contract, no declaration — the probe must still report the failure."""

    synthetic_forward_probe_kwargs = None  # type: ignore[assignment]


@pytest.fixture
def _patch_registry_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    from spectramr.models import registry as registry_mod

    fake = {
        "bool_skip_identity": _BoolSkipIdentityModel,
        "timestep_capture": _TimestepCaptureModel,
        "optional_timestep": _OptionalTimestepModel,
        "kwargs_contract": _KwargsContractModel,
        "kwargs_contract_no_hook": _KwargsContractModelNoHook,
    }

    def _fake_get_model_class(name: str) -> Any:
        return fake[name]

    monkeypatch.setattr(registry_mod, "get_model_class", _fake_get_model_class)


def test_probe_tolerates_bool_skip_flag(_patch_registry_audit: None) -> None:
    """A ``bool`` skip flag must not crash the audit; True == skip."""
    cfg = _cfg(model_type="bool_skip_identity")
    result = synthetic_forward_probe(cfg, device="cpu")
    # Identity-collapse check is skipped (not raised), so the probe passes.
    assert result.passed
    assert result.category != "identity_collapse"


def test_probe_uses_mid_schedule_timestep(_patch_registry_audit: None) -> None:
    """The probe must exercise a non-trivial mid-schedule timestep, not t=0
    (the fully-denoised boundary that hides time-embedding bugs)."""
    _CAPTURED_TIMESTEP.clear()
    cfg = _cfg(model_type="timestep_capture")
    # Provide a discoverable schedule length so the probe can pick the middle.
    #
    # `timesteps`, the CANONICAL spelling. This stub said `num_timesteps` — the
    # retired one — and so did the reader, so the two agreed and this test stayed
    # green while every real arm was probed at t=1: no schema carries
    # `num_timesteps`, so `_resolve_probe_timestep` always hit its fallback. A
    # stand-in that spells a knob the way no real config can is not a weaker
    # test, it is a test of the wrong thing (see the same hazard recorded in
    # `tests/utils/data_config_stub.py`).
    cfg.training = types.SimpleNamespace(diffusion=types.SimpleNamespace(timesteps=20))
    synthetic_forward_probe(cfg, device="cpu")

    t = _CAPTURED_TIMESTEP.get("t")
    assert t is not None, "probe never filled the required `timestep` arg"
    assert int(t.min()) > 0, "probe still feeds the trivial t=0 boundary"
    assert int(t.max()) == 10, "expected mid-schedule t = num_timesteps // 2"


def test_the_schedule_is_also_read_from_model_kwargs(
    _patch_registry_audit: None,
) -> None:
    """Third blind spot with the same symptom as the two retired spellings.

    ``training.diffusion.timesteps`` is not the only legitimate home for the
    schedule: the whole k-space cold-diffusion cohort declares it in
    ``model.model_kwargs`` and carries no ``training.diffusion`` block at all.
    ``experiment_11_attention_none`` declares 28 there and was probed at t=1
    — the fallback — so the "mid-schedule" claim was still false for it after
    both retired spellings were fixed.
    """
    _CAPTURED_TIMESTEP.clear()
    cfg = _cfg(model_type="timestep_capture", model_kwargs={"timesteps": 28})
    synthetic_forward_probe(cfg, device="cpu")

    t = _CAPTURED_TIMESTEP.get("t")
    assert t is not None
    assert int(t.max()) == 14, (
        "expected mid-schedule t = model_kwargs.timesteps // 2, got "
        f"{int(t.max())} (1 means the fallback fired, i.e. model_kwargs is "
        "still invisible to the resolver)"
    )


def test_the_strategy_path_wins_over_model_kwargs(_patch_registry_audit: None) -> None:
    """Precedence is stated, not incidental: the strategy schedule is consulted
    first, because that is the one the training loop samples ``t`` from."""
    _CAPTURED_TIMESTEP.clear()
    cfg = _cfg(model_type="timestep_capture", model_kwargs={"timesteps": 28})
    cfg.training = types.SimpleNamespace(diffusion=types.SimpleNamespace(timesteps=20))
    synthetic_forward_probe(cfg, device="cpu")
    assert int(_CAPTURED_TIMESTEP["t"].max()) == 10


def test_an_optional_timesteps_is_filled_when_a_schedule_is_declared(
    _patch_registry_audit: None,
) -> None:
    """Sensitivity pair, half 1: a default no longer means "leave it None".

    The probe filled only parameters with NO default, and every real diffusion
    forward makes ``timesteps`` optional — so the probe passed ``None`` and the
    model degraded to t=0.
    """
    _CAPTURED_TIMESTEP.clear()
    cfg = _cfg(model_type="optional_timestep", model_kwargs={"timesteps": 28})
    result = synthetic_forward_probe(cfg, device="cpu")
    assert result.passed, result.message

    t = _CAPTURED_TIMESTEP.get("t")
    assert t is not None, "probe left the optional `timesteps` at None"
    assert int(t.max()) == 14


def test_an_optional_timesteps_stays_none_when_no_schedule_is_declared(
    _patch_registry_audit: None,
) -> None:
    """Sensitivity pair, half 2: gated on the config, never unconditional.

    A model whose ``timesteps=None`` selects a genuinely different mode must keep
    receiving ``None`` when the arm declares no diffusion schedule. Without this
    half, the fix above would be indistinguishable from "always inject a
    timestep", which is a behaviour change for every non-diffusion model whose
    forward happens to accept one.
    """
    _CAPTURED_TIMESTEP.clear()
    cfg = _cfg(model_type="optional_timestep")  # no schedule anywhere
    result = synthetic_forward_probe(cfg, device="cpu")
    assert result.passed, result.message
    assert _CAPTURED_TIMESTEP.get("t") is None


def test_the_probe_consults_the_model_declared_kwargs_contract(
    _patch_registry_audit: None,
) -> None:
    """Sensitivity pair, half 1: a ``**kwargs`` contract is reachable when declared.

    ``inspect.signature`` skips ``VAR_KEYWORD`` by construction, so no amount of
    introspection can discover this. The model declares it instead.
    """
    cfg = _cfg(model_type="kwargs_contract")
    result = synthetic_forward_probe(cfg, device="cpu")
    assert result.passed, f"{result.category}: {result.message}"


def test_without_the_declaration_the_probe_reports_the_models_own_message(
    _patch_registry_audit: None,
) -> None:
    """Sensitivity pair, half 2: the failure is still reported, and reported RIGHT.

    The hook must not become a way to make failures disappear. An undeclared
    contract still fails Tier 2 — and the ``fix_hint`` must carry the model's own
    diagnosis rather than blaming ``patch_size``, which is what sent the
    experiment_11 audit to inspect shape knobs over a physics-wiring ValueError.
    """
    cfg = _cfg(model_type="kwargs_contract_no_hook")
    result = synthetic_forward_probe(cfg, device="cpu")
    assert not result.passed
    assert "ValueError" in result.message
    assert "measured" in result.message
    assert "patch_size" not in (result.fix_hint or ""), (
        "the hint still blames patch_size for a contract raise: "
        f"{result.fix_hint!r}"
    )
    assert "synthetic_forward_probe_kwargs" in (result.fix_hint or "")


def test_signature_derived_values_win_over_the_hook(
    _patch_registry_audit: None,
) -> None:
    """The hook fills gaps; it does not override config-aware resolution.

    ``_KwargsContractModel``'s hook returns ``timesteps=999``, an out-of-range
    sentinel. Its forward takes ``timesteps`` only through ``**kwargs``, so here
    the hook legitimately supplies it — but the moment a model lists the
    parameter in its signature, ``_resolve_probe_timestep``'s config-derived
    value must not be clobbered by a model-side guess.
    """
    _CAPTURED_TIMESTEP.clear()
    cfg = _cfg(model_type="kwargs_contract")
    synthetic_forward_probe(cfg, device="cpu")
    # No signature entry for `timesteps` here, so the hook's value is used.
    assert int(_CAPTURED_TIMESTEP["hook_t"].max()) == 999

    _CAPTURED_TIMESTEP.clear()
    cfg = _cfg(model_type="optional_timestep", model_kwargs={"timesteps": 28})
    synthetic_forward_probe(cfg, device="cpu")
    # Signature entry present -> config-derived mid-schedule, not a hook guess.
    assert int(_CAPTURED_TIMESTEP["t"].max()) == 14


# ── dead_loss gate honors synthetic_forward_probe_skip ─────────────────
# Self-computing-strategy models (Fisher-Rao geodesic, McCann ICNN) compute the
# real objective OUTSIDE the configured LossComputer; their YAML image_losses is a
# placeholder. Several are identity-at-initialisation (an ICNN / Euclidean geodesic
# endpoint returns ~input), so the probe's configured l1 == 0 (output == phantom
# target) and the dead_loss gate FALSE-POSITIVES — even though the arm trains on
# the cluster (b39_mccann_path: g_total_loss=0.0758). The gate must honor the same
# {"dead_loss"} opt-out the identity_collapse / input_invariant checks already do.


class _DeadLossIdentityModel(torch.nn.Module):
    """Identity-at-init model mirroring the geodesic generators: it declares the
    ``identity_collapse`` opt-out (output ~= input by design) AND ``dead_loss``
    (the configured placeholder l1 is ~0 because output ~= the phantom target)."""

    synthetic_forward_probe_skip: ClassVar[set[str]] = {"identity_collapse", "dead_loss"}

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _DeadLossIdentityModelNoSkip(_DeadLossIdentityModel):
    """Same identity model WITHOUT the dead_loss opt-out (only identity_collapse,
    so it still reaches the gate) — the gate must fire."""

    synthetic_forward_probe_skip: ClassVar[set[str]] = {"identity_collapse"}


def _cfg_with_image_l1(model_type: str) -> Any:
    """A probe config whose declarative image loss is a warmup-off l1 — so the
    only thing keeping the configured total non-zero is the model output differing
    from the phantom target (an identity model drives it to exactly 0)."""
    cfg = _cfg(model_type=model_type)
    # `losses.output_domain` -> `losses.policy.output_domain`. With the flat
    # stand-in the probe read None, so it never resolved the image loss and the
    # dead_loss gate could not fire -- the control test asserting it DOES fire
    # was the one that noticed.
    cfg.losses = block_stub(
        "losses",
        output_domain="image",
        image_losses=[{"name": "l1", "weight": 1.0}],
        reconstruction=types.SimpleNamespace(warmup_iterations=0),
    )
    return cfg


def test_dead_loss_gate_fires_for_identity_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: an identity model (output == phantom target) drives the configured
    l1 to 0, so the dead_loss gate fires when the model does NOT opt out."""
    from spectramr.models import registry as registry_mod

    monkeypatch.setattr(
        registry_mod, "get_model_class", lambda _name: _DeadLossIdentityModelNoSkip
    )
    result = synthetic_forward_probe(_cfg_with_image_l1("dead_loss_noskip"), device="cpu")
    assert result.category == "dead_loss", (
        "identity model with a configured l1 should trip the dead_loss gate"
    )


def test_dead_loss_gate_honors_probe_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SAME identity model, but declaring ``{"dead_loss"}``, must NOT be
    flagged dead_loss — matching the cluster reality for the geodesic arms."""
    from spectramr.models import registry as registry_mod

    monkeypatch.setattr(
        registry_mod, "get_model_class", lambda _name: _DeadLossIdentityModel
    )
    result = synthetic_forward_probe(_cfg_with_image_l1("dead_loss_skip"), device="cpu")
    assert result.category != "dead_loss", (
        "dead_loss gate ignored synthetic_forward_probe_skip={'dead_loss'}"
    )


# ── the probe builds the model TRAINING builds ─────────────────────────────
#
# Every other test in this file hands the probe a test double through a
# patched registry, so they all exercise the direct-construction branch. The
# branch production takes -- a registered model, built through the factory --
# had no coverage at all, which is precisely how the divergence below survived:
# `audit --probe` assembled its own constructor kwargs and skipped three
# contract-gated SSOT injections that training performs, so an arm could pass
# the probe and still diverge in training.


@_register_model("_probe_ssot_witness", "reconstruction")
class _ProbeSSOTWitness(torch.nn.Module):
    """Records the SSOT kwargs it was constructed with."""

    seen: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        acceleration_config: Any = None,
        kspace_log_scaled: Any = None,
        use_dc: Any = None,
        dc_method: Any = None,
        dc_weight: Any = None,
        **kwargs: Any,
    ):
        super().__init__()
        type(self).seen = {
            "acceleration_config": acceleration_config,
            "kspace_log_scaled": kspace_log_scaled,
            "use_dc": use_dc,
            "dc_method": dc_method,
            "dc_weight": dc_weight,
        }
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _run_ssot_probe(config) -> dict[str, Any]:
    """Run the probe and return what the constructed model saw.

    Asserts the probe itself succeeded first, so a probe failure reports its own
    message instead of surfacing as an empty ``seen`` dict.
    """
    _ProbeSSOTWitness.seen = {}
    result = synthetic_forward_probe(
        config, device="cpu", backward=False, use_phantom=False
    )
    assert result.passed, f"probe failed ({result.category}): {result.message}"
    return _ProbeSSOTWitness.seen


def _ssot_witness_config(log_scaling: bool = True):
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings(
        model={
            "model_type": "_probe_ssot_witness",
            "in_channels": 1,
            "out_channels": 1,
        },
        data={
            "sampling": {"patch_size": [16, 16]},
            "loader": {"batch_size": 1},
            "processing": {"enable_log_scaling": log_scaling},
        },
        undersampling={"enabled": True},
        physics={
            "data_consistency": {"enabled": True, "method": "soft", "weight": 0.5}
        },
        optimization={},
        logging={},
    )


class TestProbeConstructsTheModelTrainingBuilds:
    """The probe's constructor kwargs must be the training path's kwargs."""

    def test_acceleration_config_reaches_the_probed_model(self) -> None:
        """The silent half of #1306: every kspace_filling arm declares
        ``undersampling:`` and none of it used to reach the probed model."""
        seen = _run_ssot_probe(_ssot_witness_config())
        assert seen.get("acceleration_config") is not None, (
            "audit --probe built the model without acceleration_config, which "
            "training injects — the probe validates a model training does not "
            "build."
        )

    def test_kspace_log_scaled_reaches_the_probed_model(self) -> None:
        """#1281: the magnitude ceiling enforces a PHYSICAL ratio, so it must
        know whether the k-space it bounds is log1p-compressed."""
        seen = _run_ssot_probe(_ssot_witness_config(log_scaling=True))
        assert seen.get("kspace_log_scaled") is True

    def test_the_declared_value_is_the_one_that_lands(self) -> None:
        """Negative control: a hardcoded injection would pass the test above."""
        seen = _run_ssot_probe(_ssot_witness_config(log_scaling=False))
        assert seen.get("kspace_log_scaled") is False

    def test_data_consistency_reaches_the_probed_model(self) -> None:
        seen = _run_ssot_probe(_ssot_witness_config())
        assert seen.get("use_dc") is True
        assert seen.get("dc_method") == "soft"
        assert seen.get("dc_weight") == 0.5

    def test_a_conflicting_model_kwargs_copy_is_reported_not_raised(self) -> None:
        """The probe's tolerance contract still holds for the new failure mode:
        a config conflict is a structured ProbeResult, never an exception that
        would crash the audit CLI."""
        config = _ssot_witness_config(log_scaling=False)
        object.__setattr__(
            config.model, "model_kwargs", {"kspace_log_scaled": True}
        )
        result = synthetic_forward_probe(
            config, device="cpu", backward=False, use_phantom=False
        )
        assert isinstance(result, ProbeResult)
        assert result.passed is False
        assert "kspace_log_scaled" in result.message


# ── Probe measurement width (#1346) ────────────────────────────────────
#
# The probe inflates its forward input to ``2 * in_channels`` for models that
# declare the S-maps concat, because the backbone's first conv is built at that
# width. The measurement handed to ``synthetic_forward_probe_kwargs`` must NOT
# ride along: the training strategy sets ``kspace_measured = input_batch``,
# which is at the declared ``in_channels``.
#
# The gap was invisible for a year because the generator's own DC path narrows
# a too-wide measurement before using it. Only a backbone with an INTERNAL
# ``MaskedReplacementDataConsistency`` -- which receives the kwarg verbatim -- can see it,
# so ``_MeasurementWidthModel`` raises on the mismatch the way that DC does.
# These assert at the CALL SITE (through ``synthetic_forward_probe``), not on a
# helper: a helper-only pin would score the real defect green.


class _MeasurementWidthModel(torch.nn.Module):
    """Miniature of the internal-DC arm that exposed #1346."""

    seen_hook_width: int | None = None
    seen_forward_width: int | None = None

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Read by ``model_expects_smaps_concat`` -> the probe inflates.
        self.expects_smaps_concat = True
        self.conv = torch.nn.Conv2d(in_channels * 2, out_channels, 3, padding=1)

    def synthetic_forward_probe_kwargs(self, x: torch.Tensor) -> dict:
        type(self).seen_hook_width = int(x.shape[1])
        return {"kspace_measured": x.detach().clone()}

    def forward(
        self, x: torch.Tensor, kspace_measured: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        type(self).seen_forward_width = int(x.shape[1])
        out = self.conv(x)
        if kspace_measured is not None and kspace_measured.shape[1] != out.shape[1]:
            # Stands in for MaskedReplacementDataConsistency's broadcast against the
            # prediction -- the crash swin_diff_rec died on.
            raise RuntimeError(
                f"measurement width {kspace_measured.shape[1]} does not match "
                f"prediction width {out.shape[1]}"
            )
        return out


class _NoConcatMeasurementModel(_MeasurementWidthModel):
    """Same hook, but declares no concat -- the probe must not inflate."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.expects_smaps_concat = False
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1)


class _ExplicitExpansionModel(_MeasurementWidthModel):
    """Declares its own exact probe width -- must be handed through unsliced."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, **kwargs: Any):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        del self.expects_smaps_concat
        self.condition_with_smaps = False
        self.synthetic_forward_probe_input_channels = in_channels * 3
        self.conv = torch.nn.Conv2d(in_channels * 3, out_channels, 3, padding=1)

    def forward(
        self, x: torch.Tensor, kspace_measured: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        type(self).seen_forward_width = int(x.shape[1])
        return self.conv(x)


def test_probe_hook_gets_declared_width_while_input_stays_inflated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1346: the measurement is narrowed, the forward input is not.

    Two shapes of the same rule, per CLAUDE.md #15 -- asserting only the hook
    width would pass a "fix" that stopped inflating altogether and broke every
    concat arm's first conv.
    """
    _MeasurementWidthModel.seen_hook_width = None
    _MeasurementWidthModel.seen_forward_width = None
    _patch_one(monkeypatch, "meas_width", _MeasurementWidthModel)
    cfg = _cfg(model_type="meas_width", in_channels=2, out_channels=2, patch_size=[8, 8])

    result = synthetic_forward_probe(cfg, device="cpu", backward=False)

    assert result.passed, result.message
    assert _MeasurementWidthModel.seen_hook_width == 2
    assert _MeasurementWidthModel.seen_forward_width == 4


def test_probe_hook_input_untouched_without_smaps_concat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No inflation -> no narrowing. Guards against over-slicing."""
    _NoConcatMeasurementModel.seen_hook_width = None
    _NoConcatMeasurementModel.seen_forward_width = None
    _patch_one(monkeypatch, "no_concat", _NoConcatMeasurementModel)
    cfg = _cfg(model_type="no_concat", in_channels=2, out_channels=2, patch_size=[8, 8])

    result = synthetic_forward_probe(cfg, device="cpu", backward=False)

    assert result.passed, result.message
    assert _NoConcatMeasurementModel.seen_hook_width == 2
    assert _NoConcatMeasurementModel.seen_forward_width == 2


def test_probe_hook_not_sliced_for_explicitly_declared_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``synthetic_forward_probe_input_channels`` is the model's own
    declaration; the narrowing keys on the inflation THIS function performed,
    so an escape-hatch width must arrive whole."""
    _ExplicitExpansionModel.seen_hook_width = None
    _ExplicitExpansionModel.seen_forward_width = None
    _patch_one(monkeypatch, "explicit_exp", _ExplicitExpansionModel)
    cfg = _cfg(model_type="explicit_exp", in_channels=2, out_channels=2, patch_size=[8, 8])

    result = synthetic_forward_probe(cfg, device="cpu", backward=False)

    assert result.passed, result.message
    assert _ExplicitExpansionModel.seen_forward_width == 6
    assert _ExplicitExpansionModel.seen_hook_width == 6


def test_probe_resolves_generator_kwargs_on_its_own_device() -> None:
    """``audit --probe`` must build the model training builds, on the probe's
    device (#1508).

    This module exists so the probe and the training builders resolve
    constructor kwargs from ONE place; a device injected on the training path
    but not here would reintroduce exactly the divergence it removed -- a probe
    exercising the CPU mask path while training exercises the table path.

    Checked structurally on the call site rather than by string search: a
    ``getsource``/``in`` assertion is satisfied by a comment or a docstring
    mentioning the keyword (#1501), and this one is not.
    """
    import ast
    import inspect

    from spectramr.infrastructure.validation import forward_probe

    tree = ast.parse(inspect.getsource(forward_probe))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_full_generator_kwargs"
    ]
    assert calls, "the probe no longer calls the shared kwarg-resolution SSOT"
    for call in calls:
        keywords = {kw.arg for kw in call.keywords}
        assert "device" in keywords, (
            "resolve_full_generator_kwargs is called without device=; the probe "
            "would construct the model on a different device than training does"
        )


def test_the_probe_builds_its_losses_on_the_probes_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_configured_loss_total`` must move each loss onto the probe's device.

    ``create_loss`` hands back a module on CPU while ``pred``/``target`` live on
    the probe's device, so a loss carrying a parameter or buffer -- ``perceptual``
    (VGG19 plus its ImageNet mean/std), ``lpips``, ``dino_perceptual`` -- raised a
    device ``RuntimeError`` inside the computer. The computer re-raises only
    ``ValueError``; every other exception is downgraded to ``logger.warning`` and
    the term is dropped, so the probe scored the arm on a partial objective and
    still exited 0.

    The failure needs pred/target OFF the CPU, which is why no CPU-only run ever
    saw it. ``meta`` gives that without a GPU: it is a real non-CPU device, and a
    module left on CPU compares unequal to it exactly as ``cuda:0`` did.
    """
    import spectramr.models.losses.computers.unified_diffusion_reconstruction as _computers
    import spectramr.models.losses.registry as _loss_registry
    from spectramr.infrastructure.validation.forward_probe import _configured_loss_total

    class _BufferLoss(torch.nn.Module):
        """Stands in for any loss with state -- the buffer is the thing that moves."""

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("imagenet_mean", torch.zeros(1))

    captured: dict[str, Any] = {}

    class _SpyComputer:
        def __init__(self, config: Any, device: Any) -> None:
            self.device = device

        def compute(self, **kwargs: Any) -> Any:
            captured["losses"] = kwargs["losses_dict"]
            return types.SimpleNamespace(total=torch.zeros(1, requires_grad=True))

    monkeypatch.setattr(_loss_registry, "create_loss", lambda name, **kw: _BufferLoss())
    monkeypatch.setattr(_computers, "UnifiedReconstructionLossComputer", _SpyComputer)

    config = types.SimpleNamespace(
        losses=types.SimpleNamespace(
            image_losses=[types.SimpleNamespace(name="stub_loss", enabled=True)],
            policy=types.SimpleNamespace(output_domain="image"),
        )
    )
    y = torch.zeros(1, 1, 4, 4, device="meta")

    _configured_loss_total(config, y, y)

    # `_configured_loss_total` swallows every exception into _LOSS_PROBE_SKIP, so
    # a stub that never reached the computer would leave this test vacuously green.
    assert captured, "the probe never reached the loss computer -- test is vacuous"
    for name, module in captured["losses"].items():
        for buf in module.buffers():
            assert buf.device == y.device, (
                f"loss {name!r} was handed to the computer on {buf.device} while "
                f"pred/target are on {y.device}; the computer will drop the term "
                f"with a warning and the probe will score a partial objective"
            )
