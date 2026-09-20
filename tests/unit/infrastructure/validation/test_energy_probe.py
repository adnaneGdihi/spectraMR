"""CPU-only unit tests for the attention energy probe.

All tests run on tiny hand-built ``nn.Module``s (no real model, no GPU), so they
never touch the ``pipeline="probe"`` accelerator contract. They pin the energy
math, the hook lifecycle, the OOD construction, and the ranking.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tests.utils.data_config_stub import DataConfigStub

torch = pytest.importorskip("torch")
nn = torch.nn

from spectramr.infrastructure.validation import energy_probe as ep  # noqa: E402


# ── toy modules ───────────────────────────────────────────────────────────────
class _Gain(nn.Module):
    """Attention stub with a fixed energy gain ``k`` (rho == k)."""

    def __init__(self, k: float) -> None:
        super().__init__()
        self.k = float(k)

    def forward(self, x, *_, **__):
        return x * self.k


class _MagGain(nn.Module):
    """Input-magnitude-dependent gain: rho grows with ||x|| (unstable)."""

    def forward(self, x, *_, **__):
        return x * x.abs().mean()


class _Block(nn.Module):
    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.attention = attention

    def forward(self, x, t=None):
        return self.attention(x)


class _ToyGen(nn.Module):
    """Minimal generator: one hooked ``downs.0.attention`` + a toy ``sample``."""

    def __init__(self, attention: nn.Module, *, out_scale: float = 1.0, n_steps: int = 4) -> None:
        super().__init__()
        self.downs = nn.ModuleList([_Block(attention)])
        self.out_scale = float(out_scale)
        self.n_steps = int(n_steps)
        self.condition_with_smaps = False

    def forward(self, x, timesteps=None, **__):
        for b in self.downs:
            x = b(x)
        return x * self.out_scale

    def sample(self, measurement, mask=None, **__):
        x = measurement
        for _ in range(self.n_steps):
            x = self(x)  # __call__ so the top-level forward hook fires per step
        return x


def _x(b=1, c=2, h=8, w=8):
    torch.manual_seed(0)
    return torch.randn(b, c, h, w)


# ── energy primitives ─────────────────────────────────────────────────────────
def test_rho_of_known_gain_and_identity():
    x = _x()
    assert ep._energy(x * 3.0) == pytest.approx(3.0 * ep._energy(x), rel=1e-5)
    assert ep._rho(ep._energy(x), ep._energy(x * 3.0)) == pytest.approx(3.0, rel=1e-5)
    assert ep._rho(ep._energy(x), ep._energy(x)) == pytest.approx(1.0, rel=1e-6)


def test_rho_guards_zero_input():
    assert math.isfinite(ep._rho(0.0, 1.0))  # eps floor, no div-by-zero


def test_principal_tensor_unwraps_tuple_and_dict():
    t = torch.ones(2)
    assert ep._principal_tensor((t, 5)) is t
    assert ep._principal_tensor({"a": t}) is t
    assert ep._principal_tensor(t) is t
    assert ep._principal_tensor(7) is None


# ── hooks ─────────────────────────────────────────────────────────────────────
def test_hooks_record_known_gain_and_clean_up():
    model = _ToyGen(_Gain(2.0))
    handles, records = ep.register_energy_hooks(model)
    try:
        assert list(records) == ["downs.0.attention"]
        x = _x()
        model(x)
        model(x)
        model(x)
        recs = records["downs.0.attention"]
        assert len(recs) == 3  # append-only: one per forward call
        assert recs[0].rho == pytest.approx(2.0, rel=1e-5)
        assert recs[0].kind == "_Gain"
    finally:
        for h in handles:
            h.remove()
    model(_x())
    assert len(records["downs.0.attention"]) == 3  # removed hooks add nothing


def test_hooks_skip_identity():
    model = _ToyGen(nn.Identity())
    handles, records = ep.register_energy_hooks(model)
    for h in handles:
        h.remove()
    assert records == {}  # the attention_type: none arm hooks nothing


# ── OOD construction ──────────────────────────────────────────────────────────
def test_build_ood_scale_multiplies_norm():
    x = _x()
    y = ep.build_ood_batch(x, scale=2.0, phase_perturb=False)
    assert ep._energy(y) == pytest.approx(2.0 * ep._energy(x), rel=1e-5)


def test_build_ood_phase_preserves_magnitude_but_changes_tensor():
    x = _x(c=4)  # two complex pairs
    y = ep.build_ood_batch(x, scale=1.0, phase_perturb=True, seed=1)
    assert ep._energy(y) == pytest.approx(ep._energy(x), rel=1e-4)  # rotation preserves norm
    assert not torch.allclose(y, x)
    y2 = ep.build_ood_batch(x, scale=1.0, phase_perturb=True, seed=1)
    assert torch.allclose(y, y2)  # deterministic given seed


# ── measurement A + growth signature ──────────────────────────────────────────
def test_measure_forward_energy_shapes_and_gain():
    model = _ToyGen(_Gain(1.5))
    scales = [1.0, 2.0, 4.0]
    per_scale = ep.measure_forward_energy(
        model, _x(), scales=scales, phase_perturb=False, timestep=14
    )
    assert len(per_scale) == 3
    for s in per_scale:
        assert s.max_rho == pytest.approx(1.5, rel=1e-4)  # gain is magnitude-invariant
        assert len(s.blocks) == 1


def test_rho_growth_flags_input_dependent_gain():
    unstable = ep.measure_forward_energy(
        _ToyGen(_MagGain()), _x(), scales=[1.0, 4.0], phase_perturb=False, timestep=0
    )
    _, growth_unstable = ep._worst_and_growth(unstable)
    assert growth_unstable is not None and growth_unstable > 1.5  # gain grew with scale

    stable = ep.measure_forward_energy(
        _ToyGen(_Gain(0.9)), _x(), scales=[1.0, 4.0], phase_perturb=False, timestep=0
    )
    worst_stable, growth_stable = ep._worst_and_growth(stable)
    assert worst_stable == pytest.approx(0.9, rel=1e-4)
    assert growth_stable == pytest.approx(1.0, rel=1e-4)


# ── measurement B: trajectory slicing on a toy ────────────────────────────────
def test_measure_trajectory_slices_per_step():
    model = _ToyGen(_Gain(1.2), n_steps=5)
    config = SimpleNamespace(
        model=SimpleNamespace(in_channels=2, out_channels=1, model_kwargs={}),
        data=DataConfigStub(patch_size=[8, 8], batch_size=1),
        training=SimpleNamespace(diffusion=SimpleNamespace(sampling_steps=5, timesteps=28)),
    )
    steps = ep.measure_trajectory_energy(model, config, device="cpu")
    assert len(steps) == 5  # one StepEnergy per reverse step
    assert steps[0].step == 0
    assert steps[0].max_rho == pytest.approx(1.2, rel=1e-4)
    assert all(math.isfinite(s.model_output_norm) for s in steps)


# ── ranking ───────────────────────────────────────────────────────────────────
def _report(name, worst, growth):
    return ep.ArmEnergyReport(
        arm_name=name,
        attention_type="x",
        model_type="m",
        device="cpu",
        per_scale=(),
        trajectory=None,
        worst_rho=worst,
        rho_growth=growth,
    )


def test_rank_arms_orders_unstable_first_and_none_last():
    reports = [
        _report("stable", 0.9, 1.0),
        _report("kernelized", 3.0, 2.4),
        _report("none", None, None),
        _report("kan", 1.4, 1.3),
    ]
    order = [r.arm_name for r in ep.rank_arms(reports)]
    assert order == ["kernelized", "kan", "stable", "none"]


# ── build helpers ─────────────────────────────────────────────────────────────
def test_build_probe_batch_doubles_width_iff_smaps():
    config = SimpleNamespace(
        model=SimpleNamespace(in_channels=2, out_channels=1, model_kwargs={}),
        data=DataConfigStub(patch_size=[8, 8], batch_size=1),
    )
    plain = _ToyGen(nn.Identity())  # condition_with_smaps = False
    assert ep.build_probe_batch(plain, config, "cpu").shape[1] == 2

    smapped = _ToyGen(nn.Identity())
    smapped.condition_with_smaps = True
    assert ep.build_probe_batch(smapped, config, "cpu").shape[1] == 4


def test_build_probe_model_rejects_denoising_wrapper():
    config = SimpleNamespace(
        model=SimpleNamespace(
            model_type="whatever",
            in_channels=2,
            out_channels=1,
            model_kwargs={"denoising_model": {"model_type": "unet"}},
        )
    )
    with pytest.raises(ValueError, match="denoising_model"):
        ep.build_probe_model(config, "cpu")


def test_report_to_dict_is_json_shaped():
    r = _report("a", 2.0, 1.5)
    d = r.to_dict()
    assert d["arm_name"] == "a" and d["worst_rho"] == 2.0


# ── PR #398 review regressions ────────────────────────────────────────────────
class TestPhasePerturbOnRealCohortLayout:
    """The cohort is 5-D; the phase sweep used to be a silent no-op on it."""

    def test_phase_perturb_applies_on_5d_batch(self) -> None:
        """``patch_size: [256,256,1]`` => [B,2C,H,W,D]; rotation must still happen.

        Regression: ``build_ood_batch`` bailed with a bare ``return x`` unless the
        batch was exactly 4-D, so on EVERY attention_shootout arm the records
        labelled ``phase_perturbed=True`` were byte-identical to the magnitude-only
        ones -- a facade at 2x the compute (pitfall #16).
        """
        x = torch.randn(2, 4, 8, 8, 1)
        out = ep.build_ood_batch(x, scale=1.0, phase_perturb=True, seed=1)
        assert out.shape == x.shape
        assert not torch.equal(out, x), "phase perturbation was a no-op on a 5-D batch"

    def test_phase_perturb_is_magnitude_preserving_on_5d(self) -> None:
        """A phase rotation must not change per-pair magnitude (it is a rotation)."""
        x = torch.randn(2, 4, 8, 8, 1)
        out = ep.build_ood_batch(x, scale=1.0, phase_perturb=True, seed=2)
        mag_in = (x[:, 0::2] ** 2 + x[:, 1::2] ** 2).sqrt()
        mag_out = (out[:, 0::2] ** 2 + out[:, 1::2] ** 2).sqrt()
        assert torch.allclose(mag_in, mag_out, atol=1e-5)

    def test_odd_channel_count_raises_instead_of_silently_passing_through(self) -> None:
        with pytest.raises(ValueError, match="even channel count"):
            ep.build_ood_batch(torch.randn(1, 3, 8, 8), scale=1.0, phase_perturb=True)

    def test_too_few_dims_raises(self) -> None:
        with pytest.raises(ValueError, match="B, 2C, H, W"):
            ep.build_ood_batch(torch.randn(1, 2, 8), scale=1.0, phase_perturb=True)


class TestOverflowRanksAsMostUnstable:
    """A blown-up arm must not be reported as the safest one."""

    def test_non_finite_rho_yields_infinite_worst(self) -> None:
        """Regression: non-finite scales were FILTERED, leaving ``worst=None``.

        ``None`` reads as "no attention" downstream, so an arm whose attention
        overflowed -- the exact pathology this probe ranks -- sorted last.
        """
        per_scale = (
            ep.ScaleEnergy(
                scale=1.0, phase_perturbed=False, blocks=(), max_rho=math.inf, output_norm=0.0
            ),
            ep.ScaleEnergy(
                scale=2.0, phase_perturbed=False, blocks=(), max_rho=math.inf, output_norm=0.0
            ),
        )
        worst, growth = ep._worst_and_growth(per_scale)
        assert worst == math.inf
        assert growth == math.inf

    def test_partial_overflow_still_dominates(self) -> None:
        """One bad scale is enough -- it must not be averaged/filtered away."""
        per_scale = (
            ep.ScaleEnergy(
                scale=1.0, phase_perturbed=False, blocks=(), max_rho=1.1, output_norm=0.0
            ),
            ep.ScaleEnergy(
                scale=4.0, phase_perturbed=False, blocks=(), max_rho=math.nan, output_norm=0.0
            ),
        )
        worst, _ = ep._worst_and_growth(per_scale)
        assert worst == math.inf

    def test_overflowed_arm_outranks_a_merely_amplifying_one(self) -> None:
        def _r(name, worst, growth):
            return ep.ArmEnergyReport(
                arm_name=name,
                attention_type="kernelized",
                model_type="kspace_cold_diffusion",
                device="cpu",
                per_scale=(),
                trajectory=None,
                worst_rho=worst,
                rho_growth=growth,
            )

        blown = _r("blowup", math.inf, math.inf)
        amp = _r("amplifying", 3.5, 1.2)
        none_arm = ep.ArmEnergyReport(
            arm_name="no_attn",
            attention_type="none",
            model_type="kspace_cold_diffusion",
            device="cpu",
            per_scale=(),
            trajectory=None,
            worst_rho=None,
            rho_growth=None,
        )
        order = [r.arm_name for r in ep.rank_arms([amp, none_arm, blown])]
        assert order == ["blowup", "amplifying", "no_attn"], order


class TestProbeBatchIsActuallyKSpace:
    """The measured operating point must be k-space, and must have 2-D structure."""

    @staticmethod
    def _cfg(patch, in_ch=4, batch=2):
        return SimpleNamespace(
            model=SimpleNamespace(in_channels=in_ch, model_kwargs={}),
            data=DataConfigStub(patch_size=list(patch), batch_size=batch),
        )

    def test_batch_matches_requested_layout(self) -> None:
        model = SimpleNamespace(condition_with_smaps=False)
        x = ep.build_probe_batch(model, self._cfg([16, 16, 1]), "cpu")
        assert x.shape == (2, 4, 16, 16, 1)

    def test_smaps_doubles_the_width(self) -> None:
        x = ep.build_probe_batch(
            SimpleNamespace(condition_with_smaps=True), self._cfg([16, 16]), "cpu"
        )
        assert x.shape == (2, 8, 16, 16)

    def test_batch_is_center_concentrated_like_real_kspace(self) -> None:
        """Image-domain input is roughly flat; k-space concentrates at the centre.

        Regression: the probe fed the k-space model a raw image-domain phantom.
        """
        model = SimpleNamespace(condition_with_smaps=False)
        x = ep.build_probe_batch(model, self._cfg([32, 32]), "cpu")
        mag = (x[:, 0::2] ** 2 + x[:, 1::2] ** 2).sqrt()
        h = w = 32
        c = mag[:, :, h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2].sum()
        frac = float(c / mag.sum())
        # The central 4x4 is 1.56% of a 32x32 plane. A roughly-flat image-domain
        # tensor puts about that share there; centred k-space puts several times
        # more. Assert >=3x the uniform-area share -- enough to separate the two
        # domains without pinning the phantom's exact spectrum.
        area_frac = (4 * 4) / (32 * 32)
        assert frac > 3 * area_frac, (
            f"not center-concentrated: {frac:.3f} vs uniform {area_frac:.3f} "
            "(is the batch still image-domain?)"
        )

    def test_coils_are_not_copies_of_each_other(self) -> None:
        """Regression: every channel was identical => imag == real, zero coil diversity."""
        model = SimpleNamespace(condition_with_smaps=False)
        x = ep.build_probe_batch(model, self._cfg([16, 16], in_ch=4), "cpu")
        coil0 = x[0, 0:2]
        coil1 = x[0, 2:4]
        assert not torch.allclose(coil0, coil1), "coils are identical"
        # and the imaginary part must not simply mirror the real part
        assert not torch.allclose(x[0, 0], x[0, 1])

    def test_spatial_structure_is_two_dimensional(self) -> None:
        """Regression: the 5-D request hit synthetic_phantom's (B,C,D,H,W) contract,
        yielding a 1-column phantom replicated across every row (1 distinct row)."""
        model = SimpleNamespace(condition_with_smaps=False)
        x = ep.build_probe_batch(model, self._cfg([32, 32]), "cpu")
        plane = x[0, 0]
        assert torch.unique(plane, dim=0).shape[0] > 8, "rows are degenerate"
        assert torch.unique(plane, dim=1).shape[1] > 8, "columns are degenerate"

    def test_internal_dc_backbone_keeps_the_single_width(self) -> None:
        """The declaration is not the width contract (CLAUDE.md #17).

        A ``diff_varnet``-shaped generator declares ``condition_with_smaps``
        and resolves to ``expects_smaps_concat = False``, because it runs data
        consistency internally and is built at 1x. Reading the declaration here
        built a 2x probe batch for the six such arms in the corpus, so every
        energy reading described an untrained ChannelAdapter rather than the
        backbone (#1326). ``test_smaps_doubles_the_width`` above cannot catch
        this: there both flags agree.
        """
        internal_dc = SimpleNamespace(
            condition_with_smaps=True, expects_smaps_concat=False
        )
        x = ep.build_probe_batch(internal_dc, self._cfg([16, 16]), "cpu")
        assert x.shape == (2, 4, 16, 16), (
            "internal-DC backbone got a doubled probe batch; the resolver, not "
            "condition_with_smaps, decides the width"
        )

    def test_odd_width_raises(self) -> None:
        with pytest.raises(ValueError, match="even channel width"):
            ep.build_probe_batch(
                SimpleNamespace(condition_with_smaps=False),
                self._cfg([16, 16], in_ch=3),
                "cpu",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Wrapped-attention bookkeeping (issue #471)
#
# Every block is now wrapped in IdentityAtInitAttention. Naively that would make
# the probe report `IdentityAtInitAttention` for every arm -- losing the mechanism
# name the probe exists to surface -- and would hide the raw per-block gain behind
# the learned gamma. Both sites are hooked and the label is unwrapped.
# ─────────────────────────────────────────────────────────────────────────────


def test_hooks_label_the_mechanism_not_the_wrapper() -> None:
    import torch
    from torch import nn

    from spectramr.infrastructure.validation.energy_probe import register_energy_hooks
    from spectramr.models.blocks.attention import ChannelAttention, IdentityAtInitAttention

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = IdentityAtInitAttention(ChannelAttention(32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.attention(x)

    net = _Net().eval()
    handles, records = register_energy_hooks(net)
    try:
        with torch.no_grad():
            net(torch.randn(1, 32, 8, 8))
    finally:
        for h in handles:
            h.remove()

    kinds = {name: recs[0].kind for name, recs in records.items() if recs}
    assert "IdentityAtInitAttention" not in kinds.values(), (
        "the probe must name the mechanism, not the wrapper"
    )
    assert kinds["attention"] == "ChannelAttention"


def test_hooks_capture_both_effective_and_raw_gain() -> None:
    """``.attention`` is the gamma-scaled (effective) gain; ``.attention.inner`` the raw."""
    import torch
    from torch import nn

    from spectramr.infrastructure.validation.energy_probe import register_energy_hooks
    from spectramr.models.blocks.attention import ChannelAttention, IdentityAtInitAttention

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = IdentityAtInitAttention(ChannelAttention(32))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.attention(x)

    net = _Net().eval()
    handles, records = register_energy_hooks(net)
    try:
        with torch.no_grad():
            net(torch.randn(1, 32, 8, 8))
    finally:
        for h in handles:
            h.remove()

    assert "attention" in records and "attention.inner" in records
    # gamma = 0, so the effective gain is exactly 1 while the raw block attenuates.
    assert records["attention"][0].rho == pytest.approx(1.0, abs=1e-6)
    assert records["attention.inner"][0].rho < 1.0


# ── #2122: the probe resolves kwargs through the one owner ────────────────────
class _CeilingGen(nn.Module):
    """Stands in for ``kspace_cold_diffusion``'s magnitude-ceiling contract.

    Reproduces the guard that made the probe skip 12 of 12 arms in its own
    cohort: a declared ``output_kspace_clip_ratio`` cannot be bounded without
    knowing which domain it bounds (#1281), so the constructor refuses rather
    than assuming one. The refusal is correct; the probe was on the wrong side
    of it because it re-derived the kwargs instead of asking the SSOT.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_kspace_clip_ratio: float | None = None,
        kspace_log_scaled: bool | None = None,
        acceleration_config: object | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__()
        if output_kspace_clip_ratio is not None and kspace_log_scaled is None:
            raise ValueError(
                "output_kspace_clip_ratio is set but kspace_log_scaled was never "
                "supplied, so the magnitude ceiling cannot be built in the right domain."
            )
        self.kspace_log_scaled = kspace_log_scaled
        self.acceleration_config = acceleration_config
        self.seen_kwargs = kwargs
        self.body = nn.Identity()


def _ceiling_arm(*, log_scaled: bool = True) -> SimpleNamespace:
    """A config shaped like the arms that could not be probed."""
    # Routed through the stub's flat->canonical map, not assigned after the
    # fact: every sub-block is frozen=True (non-negotiable 1).
    data = DataConfigStub(patch_size=[8, 8], batch_size=1, log_scaling=log_scaled)
    return SimpleNamespace(
        model=SimpleNamespace(
            model_type="_ceiling_probe_stub",
            in_channels=2,
            out_channels=2,
            model_kwargs={"output_kspace_clip_ratio": 1.3},
        ),
        data=data,
        undersampling=SimpleNamespace(base_acceleration=1.0, max_acceleration=32.0),
    )


@pytest.fixture
def _registered_ceiling_gen(monkeypatch):
    """Route ``get_model_class`` to the stub without touching the real registry."""
    import spectramr.models.registry as registry

    monkeypatch.setattr(registry, "get_model_class", lambda name: _CeilingGen)
    return _CeilingGen


def test_build_probe_model_injects_the_log_scaling_ssot(_registered_ceiling_gen):
    """The arm that could not be built is built, and carries the injected flag.

    Reverting ``build_probe_model`` to its own kwargs derivation turns this red
    with the constructor's own ValueError -- which is exactly the message the
    cohort sweep printed 12 times before exiting (#2122).
    """
    model = ep.build_probe_model(_ceiling_arm(log_scaled=True), "cpu")
    assert model.kspace_log_scaled is True


def test_build_probe_model_reads_the_data_block_not_a_default(_registered_ceiling_gen):
    """``False`` is forwarded as ``False``, never elided into "unset"."""
    model = ep.build_probe_model(_ceiling_arm(log_scaled=False), "cpu")
    assert model.kspace_log_scaled is False


def test_build_probe_model_injects_the_acceleration_config(_registered_ceiling_gen):
    """Step 3a is part of the same resolution and must not be re-dropped."""
    config = _ceiling_arm()
    model = ep.build_probe_model(config, "cpu")
    assert model.acceleration_config is config.undersampling
