from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
import torch.nn as nn

from spectramr.models.generators.kspace_cold_diffusion_generator import (
    KSpaceColdDiffusionGenerator,
)


def _measurement_kwargs(x: torch.Tensor) -> dict:
    """The measurement kwargs a real training forward always supplies.

    ``dc_method`` defaults to ``'hard'``, so a bare ``model(x, timestep=...)``
    in train mode builds a DC layer and an output magnitude bound and then runs
    NEITHER, because both are gated on ``kspace_measured``. These tests passed
    that way for as long as the skip was silent — the mock-fed shape assertions
    hold whether or not the physics fires, so they never noticed. The generator
    now raises instead, and supplying the measurement is the faithful fix:
    production always does (``_build_generator_kwargs``), and it makes these
    exercise the path that actually runs.
    """
    mask = torch.zeros(x.shape[0], 1, *x.shape[2:])
    mask[..., ::2] = 1.0
    return {"kspace_measured": x * mask, "mask": mask}


def test_generator_dc_methods_sourced_from_physics_ssot():
    """The generator validates ``dc_method`` against the single physics SSOT
    frozenset, not a private literal copy (canonical-homes / SSOT). This keeps
    the model-internal DC builder and the reverse-diffusion sampler in lockstep
    (the 2026-07-05 ``dc_method='adaptive'`` divergence)."""
    import spectramr.models.generators.kspace_cold_diffusion_generator as kcg
    from spectramr.infrastructure.physics import data_consistency as dc

    assert kcg.VALID_DC_METHODS is dc.VALID_DC_METHODS


def test_generator_builds_noise_adaptive_dc_layer():
    """``dc_method='noise_adaptive'`` builds the Wiener NoiseAdaptiveDataConsistency
    layer (``dc_weight`` forwarded as the trust temperature β), and its trust
    telemetry is surfaced under ``noise_adaptive_trust/*`` tags."""
    from spectramr.infrastructure.physics.data_consistency import (
        NoiseAdaptiveDataConsistency,
    )

    model = KSpaceColdDiffusionGenerator(
        in_channels=2,
        out_channels=2,
        base_channels=16,
        num_layers=2,
        attention_type="none",
        dc_method="noise_adaptive",
        dc_weight=0.5,
    )
    assert isinstance(model.dc_layer, NoiseAdaptiveDataConsistency)
    assert float(model.dc_layer.beta) == pytest.approx(0.5)
    assert set(model.get_kan_trust_map_telemetry()) == {
        "noise_adaptive_trust/center",
        "noise_adaptive_trust/periphery",
        "noise_adaptive_trust/mean",
        "noise_adaptive_trust/std",
    }


class TestKSpaceColdDiffusionGenerator:
    """Unit tests for KSpaceColdDiffusionGenerator."""

    @property
    def device(self):
        return torch.device("cpu")

    def test_initialization(self):
        """Test model initialization with standard params."""
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=16,
            num_layers=2,
            attention_type="none",  # 'unet' backbone supports none/channel/spatial
        )
        assert isinstance(model, nn.Module)
        assert model.in_channels == 2

    def test_spade_kwargs_fail_loud(self):
        """F-SPADE (smoke_audit_20260524): SPADE marker-conditioning kwargs
        with no consuming backbone must raise at construction, not silently
        run as a plain backbone.

        ``experiment_11b_spade_cold_diffusion`` declared
        ``spade_hidden_channels`` / ``spade_norm_type`` in model_kwargs, but
        ``SPADEBlock``/``SPADEEncoder`` are wired into no
        kspace_cold_diffusion backbone — the kwargs were silently dropped and
        the arm ran as a plain ``complex_unet`` (a mislabeled "SPADE" arm).
        CLAUDE.md #9 forbids silent fallbacks; the guard fails loud instead.
        """
        with pytest.raises(ValueError, match="SPADE"):
            KSpaceColdDiffusionGenerator(
                in_channels=2,
                out_channels=2,
                base_channels=8,
                num_layers=2,
                spade_hidden_channels=64,
                spade_norm_type="group",
            )

    def test_no_spade_kwargs_builds_cleanly(self):
        """The F-SPADE guard must not fire for ordinary configs — only
        when the SPADE-specific kwargs are present."""
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            attention_type="none",
        )
        assert isinstance(model, nn.Module)

    def test_unet_backbone_rejects_unsupported_attention(self):
        """The reconstruction-UNet backbone (``backbone_type='unet'``, non-pure)
        builds ``ConfigurableResidualBlock``, which implements only channel/
        spatial attention. Requesting self/dual_domain there must FAIL LOUD at
        construction with an actionable message (use ``complex_unet``), not the
        generic block-level raise (pitfall #9). This is the guard for the
        attention-refactor's ``unet`` + default ``attention_type='self'``
        breakage."""
        with pytest.raises(ValueError, match="complex_unet"):
            KSpaceColdDiffusionGenerator(
                in_channels=2,
                out_channels=2,
                base_channels=8,
                num_layers=2,
                backbone_type="unet",
                attention_type="self",
            )

    def test_forward_pass_shape(self):
        """Test forward pass preserves shapes (U-Net like behavior)."""
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            # This case exercises shape/scale/conditioning behaviour, not S-map
            # conditioning, so it declares the plain contract. With the default
            # ``condition_with_smaps=True`` the backbone is built at 2x
            # in_channels and a bare ``in_channels`` stack is a width error —
            # which FourierBridgeNetwork used to absorb by building an untrained
            # ChannelAdapter (#1326) and now correctly refuses.
            condition_with_smaps=False,
            base_channels=8,
            num_layers=2,
            use_complex_conv=False,  # Test standard path first
            attention_type="none",  # 'unet' backbone supports none/channel/spatial
        )

        # Batch size 2, Channels 2, Size 32x32
        x = torch.randn(2, 2, 32, 32)
        timestep = torch.randint(0, 1000, (2,))

        output = model(x, timestep=timestep, **_measurement_kwargs(x))

        if isinstance(output, tuple):
            output = output[0]

        assert output.shape == x.shape

    def test_training_forward_exposes_pre_dc(self):
        """Training-mode forward returns (post_dc, pre_dc): the 2nd element is
        the generator's PRE-DC prediction (was None) for opt-in pre-DC fidelity
        supervision. Eval-mode still returns a single tensor."""
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            # This case exercises shape/scale/conditioning behaviour, not S-map
            # conditioning, so it declares the plain contract. With the default
            # ``condition_with_smaps=True`` the backbone is built at 2x
            # in_channels and a bare ``in_channels`` stack is a width error —
            # which FourierBridgeNetwork used to absorb by building an untrained
            # ChannelAdapter (#1326) and now correctly refuses.
            condition_with_smaps=False,
            base_channels=8,
            num_layers=2,
            use_complex_conv=False,
            attention_type="none",  # 'unet' backbone supports none/channel/spatial
        )
        x = torch.randn(2, 2, 32, 32)
        timestep = torch.randint(0, 1000, (2,))

        model.train()
        out = model(x, timestep=timestep, **_measurement_kwargs(x))
        assert isinstance(out, tuple) and len(out) == 2
        post_dc, pre_dc = out
        assert torch.is_tensor(pre_dc)  # not None
        assert pre_dc.shape == post_dc.shape

        model.eval()
        out_eval = model(x, timestep=timestep, **_measurement_kwargs(x))
        assert torch.is_tensor(out_eval)  # single tensor in eval, not a tuple

    def test_output_kspace_clip_ratio_rejects_nonpositive(self):
        """Build-time validation (pitfall #15): the scale-control ratio must be
        > 0 or null; an illegal value fails at construction, not mid-training."""
        with pytest.raises(ValueError, match="output_kspace_clip_ratio"):
            KSpaceColdDiffusionGenerator(
                in_channels=2,
                out_channels=2,
                base_channels=8,
                num_layers=2,
                use_complex_conv=False,
                attention_type="none",
                output_kspace_clip_ratio=0.0,
                kspace_log_scaled=False,
            )

    def test_output_kspace_clip_ratio_bounds_output_magnitude(self):
        """Phase-1 scale-control guard: with ``output_kspace_clip_ratio`` set the
        forward caps |output| at ``ratio × max|measured|`` per sample. The
        unnormalised k-space head has nothing else bounding its scale (the
        experiment_11 measurement-independent collapse). Non-vacuous: the same
        model with the guard OFF exceeds the ceiling on a tiny-scale measurement.
        """
        torch.manual_seed(0)
        kw = dict(
            in_channels=2,
            out_channels=2,
            # This case exercises shape/scale/conditioning behaviour, not S-map
            # conditioning, so it declares the plain contract. With the default
            # ``condition_with_smaps=True`` the backbone is built at 2x
            # in_channels and a bare ``in_channels`` stack is a width error —
            # which FourierBridgeNetwork used to absorb by building an untrained
            # ChannelAdapter (#1326) and now correctly refuses.
            condition_with_smaps=False,
            base_channels=8,
            num_layers=2,
            use_complex_conv=False,
            attention_type="none",
        )
        x = torch.randn(2, 2, 32, 32)
        t = torch.randint(0, 1000, (2,))
        measured = 0.001 * torch.randn(2, 2, 32, 32)  # tiny scale => tiny ceiling
        mask = torch.zeros(2, 2, 32, 32)
        mask[:, :, :, :16] = 1.0  # half observed => unobserved carries prediction
        ratio = 1.3
        ceil = ratio * float(measured.abs().amax())

        bounded = KSpaceColdDiffusionGenerator(
            output_kspace_clip_ratio=ratio, kspace_log_scaled=False, **kw
        )
        bounded.train()
        with torch.no_grad():
            ob = bounded(x, timestep=t, kspace_measured=measured, mask=mask)
        ob = ob[0] if isinstance(ob, tuple) else ob
        assert float(ob.abs().max()) <= ceil + 1e-4

        free = KSpaceColdDiffusionGenerator(**kw)  # guard off (default None)
        free.train()
        with torch.no_grad():
            of = free(x, timestep=t, kspace_measured=measured, mask=mask)
        of = of[0] if isinstance(of, tuple) else of
        assert float(of.abs().max()) > ceil  # unbounded head exceeds the ceiling

    def test_output_kspace_clip_ratio_skips_on_batch_mismatch(self):
        """The guard is SKIPPED (not crashed) when ``kspace_measured`` cannot
        broadcast to ``x_out``. The forward must still return finite output.

        This happens in production when a 5D (D>1) input restores x_out to
        [B,C,H,W,D] while kspace_measured stays flattened to [B*D,...]; the
        batch mismatch itself is what the guard keys on, and it is exercised
        directly here (see the note on the 5D route below).
        """
        torch.manual_seed(0)
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            # This case exercises shape/scale/conditioning behaviour, not S-map
            # conditioning, so it declares the plain contract. With the default
            # ``condition_with_smaps=True`` the backbone is built at 2x
            # in_channels and a bare ``in_channels`` stack is a width error —
            # which FourierBridgeNetwork used to absorb by building an untrained
            # ChannelAdapter (#1326) and now correctly refuses.
            condition_with_smaps=False,
            base_channels=8,
            num_layers=2,
            use_complex_conv=False,
            attention_type="none",
            output_kspace_clip_ratio=1.3,
            kspace_log_scaled=False,
        )
        model.train()
        # The mechanism under test is "kspace_measured's batch cannot broadcast
        # to x_out, so the ceiling is skipped rather than crashing" — expressed
        # here in 4D, which isolates it.
        #
        # It used to be expressed with a 5D [1, 2, 16, 16, 2] input, and that
        # route asserted almost nothing. The generator reads 5D as
        # [B, D, C, H, W] and flattens to [B*D, C, H, W], so that tensor reached
        # the backbone as SIXTEEN channels against in_channels=2 — absorbed by
        # a rebuilt (untrained) ChannelAdapter — and came back as (1, 2, 16, 2,
        # 2) from a (1, 2, 16, 16, 2) input, i.e. with its spatial dims mangled.
        # Only ``isfinite`` was asserted, so neither showed. The layout-correct
        # [1, D=2, C=2, 16, 16] does not work either: it raises inside
        # complex_conv on dev as well as here. The 5D volumetric path is filed
        # separately; this case no longer depends on it.
        x = torch.randn(1, 2, 16, 16)
        t = torch.randint(0, 1000, (1,))
        measured = 0.001 * torch.randn(2, 2, 16, 16)  # batch 2 != x batch 1
        mask = torch.ones(2, 1, 16, 16)
        with torch.no_grad():
            out = model(x, timestep=t, kspace_measured=measured, mask=mask)
        out = out[0] if isinstance(out, tuple) else out
        assert torch.isfinite(out).all()

    def test_timesteps_kwarg_binds_num_timesteps(self):
        """model_kwargs.timesteps must drive num_timesteps (the time-embedding
        max). A bare timesteps=28 previously fell into **kwargs and left
        num_timesteps=1000, collapsing the sinusoidal time embedding so FiLM
        was timestep-blind (Experiment-11 DC-blob contributor)."""
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            use_complex_conv=False,
            timesteps=28,
            attention_type="none",  # 'unet' backbone supports none/channel/spatial
        )
        assert model.num_timesteps == 28

    def test_conflicting_timesteps_and_num_timesteps_raise(self):
        """Conflicting explicit num_timesteps + timesteps must raise — no silent
        fallback (CLAUDE.md pitfall #9)."""
        with pytest.raises(ValueError, match="conflicting"):
            KSpaceColdDiffusionGenerator(
                in_channels=2,
                out_channels=2,
                base_channels=8,
                num_layers=2,
                use_complex_conv=False,
                num_timesteps=500,
                timesteps=28,
            )

    def test_forward_pass_complex_conv(self):
        """Test with complex convolutions enabled."""
        # Note: KSpaceColdDiffusionGenerator assumes input is (B, C, H, W)
        # but treats channels as Real/Imag pairs if use_complex_conv is True
        # Usually requires C to be even.

        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            # This case exercises shape/scale/conditioning behaviour, not S-map
            # conditioning, so it declares the plain contract. With the default
            # ``condition_with_smaps=True`` the backbone is built at 2x
            # in_channels and a bare ``in_channels`` stack is a width error —
            # which FourierBridgeNetwork used to absorb by building an untrained
            # ChannelAdapter (#1326) and now correctly refuses.
            condition_with_smaps=False,
            base_channels=8,
            num_layers=2,
            use_complex_conv=True,
            activation="complex",
            attention_type="none",  # 'unet' backbone supports none/channel/spatial
        )

        x = torch.randn(2, 2, 32, 32)
        timestep = torch.randint(0, 1000, (2,))

        output = model(x, timestep=timestep, **_measurement_kwargs(x))

        if isinstance(output, tuple):
            output = output[0]

        assert output.shape == x.shape
        # Just basic run check, assuming ComplexConv2d works if imported

    def test_time_embedding_integration(self):
        """Test that time embedding affects the output."""
        # Use a slightly larger model to ensure time embedding has measurable effect
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            # This case exercises shape/scale/conditioning behaviour, not S-map
            # conditioning, so it declares the plain contract. With the default
            # ``condition_with_smaps=True`` the backbone is built at 2x
            # in_channels and a bare ``in_channels`` stack is a width error —
            # which FourierBridgeNetwork used to absorb by building an untrained
            # ChannelAdapter (#1326) and now correctly refuses.
            condition_with_smaps=False,
            base_channels=32,
            time_embedding_dim=256,
            attention_type="none",  # 'unet' backbone supports none/channel/spatial
            # Declared off rather than left inert. This test asserts the BACKBONE
            # conditions on t, and hard DC would overwrite the sampled bins with
            # the same measurement at both timesteps — weakening the very
            # difference under test. Previously DC was built and then silently
            # skipped for want of a measurement, which happened to give the same
            # result for the wrong reason.
            dc_method="none",
        )

        x = torch.randn(1, 2, 32, 32)

        # Two different timesteps
        t1 = torch.tensor([100])
        t2 = torch.tensor([900])

        out1 = model(x, timesteps=t1)
        out2 = model(x, timesteps=t2)

        if isinstance(out1, tuple):
            out1 = out1[0]
        if isinstance(out2, tuple):
            out2 = out2[0]

        # Output should differ if time embedding is used
        assert not torch.allclose(out1, out2, atol=1e-7)

    def test_forward_pass_dual_domain_attention(self):
        """Test with dual-domain attention enabled.

        dual_domain is a block-level attention on the ``complex_unet`` backbone
        (the ``unet`` reconstruction backbone / ConfigurableResidualBlock has no
        such seam), matching the real experiment_11 arms.
        """
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            # This case exercises shape/scale/conditioning behaviour, not S-map
            # conditioning, so it declares the plain contract. With the default
            # ``condition_with_smaps=True`` the backbone is built at 2x
            # in_channels and a bare ``in_channels`` stack is a width error —
            # which FourierBridgeNetwork used to absorb by building an untrained
            # ChannelAdapter (#1326) and now correctly refuses.
            condition_with_smaps=False,
            base_channels=8,
            num_layers=2,
            backbone_type="complex_unet",
            attention_type="dual_domain",
            use_complex_conv=True,
        )

        x = torch.randn(2, 2, 32, 32)
        timestep = torch.randint(0, 1000, (2,))

        output = model(x, timestep=timestep, **_measurement_kwargs(x))

        if isinstance(output, tuple):
            output = output[0]

        assert output.shape == x.shape
        assert not torch.any(torch.isnan(output))


class TestKSpaceColdDiffusionDCBlobRegression:
    """Regression for the validation "DC blob" artefact (smoke triage of
    ``experiment_11_kspace_cold_diffusion``, 2026-05-26).

    Root cause: ``FourierBridgeNetwork.forward`` fed the phase-safe dual
    attention as ``phase_safe_attention(image_stacked, image_output)``, i.e. it
    passed the *raw input k-space* (``image_stacked`` — and under
    ``force_pure_kspace`` that tensor is k-space, DC spike at centre) as the
    ``x_kspace`` value/residual argument, while the backbone prediction
    (``image_output``) was only the ``x_image`` query.

    ``PhaseSafeDualAttention`` returns ``x_kspace + gamma * out`` with ``gamma``
    initialised to 0, so at init the attention output **equals ``x_kspace``** —
    the input k-space. The model therefore emitted (a scrambled copy of) the
    undersampled input k-space and *discarded the backbone output entirely*. The
    IFFT of undersampled input k-space is a bright central "DC blob", which is
    exactly what the validation ``fake`` PNGs showed — independent of training.

    The fix: refine the *backbone prediction* (``x_kspace=image_output``) using a
    genuine image-domain query (``ifft2c`` of the input when ``force_pure_kspace``).
    """

    def _build(self, attention_type="dual_domain"):
        # Mirror experiment_11 (pure k-space, complex_unet) at a tiny size.
        # out_channels must be >= 4 for the FourierBridge attention path to engage.
        return KSpaceColdDiffusionGenerator(
            in_channels=4,
            out_channels=4,
            base_channels=8,
            num_layers=2,
            attention_type=attention_type,
            use_complex_conv=True,
            activation="complex",
            backbone_type="complex_unet",
            force_pure_kspace=True,
            condition_with_smaps=False,
            num_timesteps=1000,
            time_embedding_dim=32,
        ).eval()

    @pytest.mark.parametrize(
        "attention_type",
        # ``dual_domain`` is the arm that had the passthrough bug; the others
        # share the FourierBridge attention block and must also keep the
        # backbone wired (the pre-fix code hit a silent ``except`` fallback for
        # every non-``dual_domain`` complex_unet arm because it called a
        # ``None`` ``phase_safe_attention``).
        ["dual_domain", "kan_dual_domain", "channel", "self", "none"],
    )
    def test_backbone_output_drives_prediction(self, attention_type):
        """The backbone (the U-Net doing the actual reconstruction) MUST
        influence the generator output for every attention type. Pre-fix the
        ``dual_domain`` arm discarded it via the attention passthrough: forcing
        the backbone output to two different constants left the generator output
        bit-identical (the input k-space).
        """
        model = self._build(attention_type)
        x = torch.randn(2, 4, 64, 64)
        t = torch.full((2,), 500, dtype=torch.long)

        const = {"v": 0.0}

        def _force_const(_mod, _inp, out):
            if isinstance(out, tuple):
                return (torch.full_like(out[0], const["v"]), *tuple(out[1:]))
            return torch.full_like(out, const["v"])

        handle = model.backbone.backbone.register_forward_hook(_force_const)
        try:
            with torch.no_grad():
                const["v"] = 1.0
                o1 = model(x, timesteps=t)
                o1 = o1[0] if isinstance(o1, tuple) else o1
                const["v"] = -1.0
                o2 = model(x, timesteps=t)
                o2 = o2[0] if isinstance(o2, tuple) else o2
        finally:
            handle.remove()

        delta = (o1 - o2).abs().max().item()
        assert delta > 1e-3, (
            f"[attention_type={attention_type}] Backbone output discarded "
            f"(DC-blob passthrough bug): forcing the backbone to +1 vs -1 changed "
            f"the generator output by only {delta:.2e}. The FourierBridge attention "
            "must refine the backbone prediction (x_kspace=image_output), not pass "
            "the input k-space through."
        )


class TestPhaseSafeDualAttentionResidualContract:
    """Documents the PhaseSafeDualAttention contract the generator must respect:
    the residual/value comes from ``x_kspace``; ``x_image`` only steers weights.
    """

    def test_residual_and_value_come_from_x_kspace(self):
        from spectramr.models.blocks.attention import PhaseSafeDualAttention

        attn = PhaseSafeDualAttention(in_channels=2, num_heads=1, reduction=1).eval()
        x_kspace = torch.randn(2, 4, 16, 16)
        with torch.no_grad():
            out_a = attn(x_kspace=x_kspace, x_image=torch.randn(2, 4, 16, 16))
            out_b = attn(x_kspace=x_kspace, x_image=torch.randn(2, 4, 16, 16))
        # gamma initialises to 0 -> output equals x_kspace, independent of x_image.
        assert torch.allclose(out_a, x_kspace, atol=1e-6)
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_attention_caps_tokens_to_avoid_oom(self):
        """At full image resolution the dense ``[B, N, N]`` softmax OOMs
        (256x256 -> N=65536 -> ~32 GiB; this is the crash exposed once the
        masking try/except was removed). The ``max_tokens`` cap pools q/k/v to
        a coarse grid before attention and upsamples the result, so a large map
        runs in bounded memory and preserves spatial shape. Without the cap
        this allocates a ~17 GiB ``[1, 65536, 65536]`` matrix and OOMs.
        """
        from spectramr.models.blocks.attention import PhaseSafeDualAttention

        attn = PhaseSafeDualAttention(
            in_channels=4, num_heads=1, reduction=1, max_tokens=256
        ).eval()
        x_kspace = torch.randn(1, 8, 256, 256)
        x_image = torch.randn(1, 8, 256, 256)
        with torch.no_grad():
            out = attn(x_kspace, x_image)
        assert out.shape == x_kspace.shape
        assert torch.isfinite(out).all()

    def test_attention_below_cap_uses_full_resolution(self):
        """Below the cap, attention runs at native resolution (no pooling),
        and the gamma=0 residual still yields an exact x_kspace passthrough."""
        from spectramr.models.blocks.attention import PhaseSafeDualAttention

        attn = PhaseSafeDualAttention(
            in_channels=2, num_heads=1, reduction=1, max_tokens=4096
        ).eval()
        x = torch.randn(2, 4, 16, 16)
        with torch.no_grad():
            out = attn(x, torch.randn(2, 4, 16, 16))
        assert out.shape == x.shape
        assert torch.allclose(out, x, atol=1e-6)

    def test_invalid_max_tokens_raises(self):
        from spectramr.models.blocks.attention import PhaseSafeDualAttention

        with pytest.raises(ValueError, match="max_tokens"):
            PhaseSafeDualAttention(in_channels=2, max_tokens=0)


class TestKSpaceColdDiffusionMultiStepSample:
    """The multi-step cold restoration that the 2026-06-08 opt-in validation
    knob (``validation.multistep_cold_sampling``) invokes through
    DiffusionTrainingStrategy. It must return correctly-shaped, FINITE K-SPACE
    (matching ``forward()``) so the downstream cold-branch IFFT stays correct —
    calling ``generate()`` (image output) instead would re-create the DC blob."""

    def test_sample_returns_finite_kspace_matching_measurement(self):
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=16,
            num_layers=2,
            attention_type="channel",
            process_type="cold_diffusion",
            # The reverse sampler always builds a magnitude ceiling, so unlike
            # the opt-in training bound this one always needs the domain.
            kspace_log_scaled=False,
            # Declared, not defaulted. This case asserts shape + finiteness of
            # the reverse loop; with S-map conditioning ON, sampling without
            # maps is a width error. Note that shape+finiteness is exactly what
            # could NOT see the original defect: an untrained random projection
            # returns output that is correctly shaped and finite. The
            # differential that does see it lives in
            # TestChannelAdapterWidthContract below.
            condition_with_smaps=False,
        ).eval()
        meas = torch.randn(1, 2, 32, 32)
        mask = torch.ones(1, 1, 32, 32)
        mask[..., ::2] = 0  # undersample
        with torch.no_grad():
            out = model.sample(measurement=meas, mask=mask, inference_timesteps=2)
        assert out.shape == meas.shape, (
            f"sample() must return k-space matching the measurement shape, got {tuple(out.shape)}"
        )
        assert torch.isfinite(out).all(), "sample() produced non-finite k-space"



class TestChannelAdapterWidthContract:
    """The train/validation width skew that made experiment_11_attention_none's
    R8x/R32x fakes anticorrelated with their targets (#1326 on the sampling
    paths).

    ``FourierBridgeNetwork.channel_adapter`` is ``None`` in ``__init__`` and was
    constructed ONLY inside ``forward``, rebuilt whenever the input width
    changed. Because it is created after the optimizer captured
    ``model.parameters()``, a non-Identity adapter is never trained, never
    checkpointed, and is redrawn on every flip. Training fed 16ch
    (``[kspace || smaps_k]``) and the multi-step validation sampler fed 8ch
    (bare measurement), so they flipped it against each other every validation.

    Every assertion here fails on the pre-fix generator. The pre-existing
    ``sample()`` test does not, and could not: it asserts shape and finiteness,
    and a random 1x1 projection returns output that is correctly shaped and
    finite. Only a differential — same weights, same input, twice, with a width
    flip in between — can see it.
    """

    IN_CH: ClassVar[int] = 4

    def _build(self, **over) -> KSpaceColdDiffusionGenerator:
        kw = {
            "in_channels": self.IN_CH,
            "out_channels": self.IN_CH,
            "base_channels": 8,
            "num_layers": 2,
            "backbone_type": "complex_unet",
            "attention_type": "none",
            "use_complex_conv": True,
            "timesteps": 4,
            "kspace_log_scaled": False,
        }
        kw.update(over)
        model = KSpaceColdDiffusionGenerator(**kw).eval()
        # ``HardDataConsistency.eval_noise_level`` defaults to 0.005, so the DC
        # layer perturbs the measured k-space with a fresh ``randn_like`` draw
        # on every eval forward THAT REACHES IT -- i.e. one carrying a mask and
        # a measurement, which is every real validation step. (A bare
        # ``forward(x, t)`` never touches the DC layer and is reproducible, so
        # this is easy to under-diagnose.) Measured on the layer itself at its
        # defaults: two identical eval calls differ by max|d| = 0.023, 0.69%
        # relative; exactly 0.0 when zeroed. That is a separate source of
        # validation non-reproducibility and it would mask the adapter
        # differential below, which is the thing under test. Silenced
        # explicitly rather than worked around, so the test states what it
        # controls for -- and the zeroing is load-bearing: without it
        # ``test_repeat_forwards_across_a_width_flip_are_bit_identical`` fails.
        for _, module in model.named_modules():
            if hasattr(module, "eval_noise_level"):
                module.eval_noise_level = 0.0
                module.train_noise_level = 0.0
        return model

    @staticmethod
    def _bridge(model: KSpaceColdDiffusionGenerator):
        return next(
            m for _, m in model.named_modules()
            if type(m).__name__ == "FourierBridgeNetwork"
        )

    def _smaps(self, b: int, h: int, w: int) -> torch.Tensor:
        """Image-domain complex maps, as ESPIRiT returns them."""
        return torch.randn(b, self.IN_CH // 2, h, w, dtype=torch.complex64)

    @staticmethod
    def _record_widths(bridge, seen: list[int]):
        """Record the channel count the backbone actually receives.

        This is the observation that discriminates: the pre-fix generator
        forwarded a bare ``in_channels`` stack and let the bridge coerce it, so
        assertions on the OUTPUT (shape, finiteness, equality) all held. Only
        watching the width at the boundary shows the difference.
        """

        def _hook(_m, inputs):
            seen.append(inputs[0].shape[1])
            return None  # a pre-hook returning non-None REPLACES the input

        return bridge.register_forward_pre_hook(_hook)

    def test_expects_smaps_concat_matches_the_backbone_width(self) -> None:
        """The elected owner (CLAUDE.md #17): one predicate decides both the
        backbone's width and whether a caller may concatenate."""
        model = self._build()
        assert model.expects_smaps_concat is True
        assert self._bridge(model).config.in_channels == 2 * model.in_channels

    def test_backbone_in_channels_is_published_for_the_contract(self) -> None:
        """The width the model-input contract compares against.

        ``in_channels`` is the BARE measurement -- correct for the concat gate
        in ``forward`` and for the validation sampler that re-enters un-doubled,
        wrong as a description of what the backbone is handed on the training
        path. The contract read ``in_channels`` and so reported a mismatch on
        every conditioned arm; it now reads this attribute, which must agree
        with the width the bridge was actually built at whichever way
        ``expects_smaps_concat`` falls.
        """
        conditioned = self._build()
        assert conditioned.backbone_in_channels == 2 * conditioned.in_channels
        assert (
            conditioned.backbone_in_channels
            == self._bridge(conditioned).config.in_channels
        )

        plain = self._build(condition_with_smaps=False)
        assert plain.backbone_in_channels == plain.in_channels
        assert plain.backbone_in_channels == self._bridge(plain).config.in_channels

    def test_the_contract_resolver_reads_the_published_width(self) -> None:
        """Wiring, not just presence (CLAUDE.md #16): the attribute is only a
        fix if the resolver that fires the warning actually reaches it."""
        from spectramr.infrastructure.training.model_input_contract import (
            resolve_model_in_channels,
        )

        model = self._build()
        channels, source = resolve_model_in_channels(model)
        assert source == "module.backbone_in_channels"
        assert channels == 2 * model.in_channels

        plain = self._build(condition_with_smaps=False)
        assert plain.expects_smaps_concat is False
        assert self._bridge(plain).config.in_channels == plain.in_channels

    def test_bare_width_raises_instead_of_building_an_untrained_adapter(self) -> None:
        """PLANTED VIOLATION: the exact call the multi-step validation path
        made. It used to return plausible garbage; it must now raise."""
        model = self._build()
        x = torch.randn(1, self.IN_CH, 16, 16)
        t = torch.zeros(1, dtype=torch.long)
        with pytest.raises(ValueError, match=r"built for"), torch.no_grad():
            model(x, t, **_measurement_kwargs(x))

    def test_adapter_is_never_a_trainable_free_rider(self) -> None:
        """Whatever adapter exists after a forward must be an Identity — a
        Conv2d here would carry weights no optimizer step ever reaches."""
        model = self._build()
        x = torch.randn(1, self.IN_CH, 16, 16)
        t = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            model(x, t, smaps=self._smaps(1, 16, 16), **_measurement_kwargs(x))
        adapter = self._bridge(model).channel_adapter
        inner = getattr(adapter, "adapter", adapter)
        assert isinstance(inner, nn.Identity), (
            f"channel_adapter resolved to {type(inner).__name__}; a non-Identity "
            "adapter is built inside forward(), after the optimizer captured "
            "parameters(), so it is untrained and absent from checkpoints."
        )

    def test_smaps_complete_a_bare_stack_to_the_trained_width(self) -> None:
        """The generator itself completes the stack, per reverse step, so the
        sampler need only hand over image-domain maps."""
        model = self._build()
        bridge = self._bridge(model)
        seen: list[int] = []

        def _rec(_m, inputs):
            seen.append(inputs[0].shape[1])
            return None  # a pre-hook returning non-None REPLACES the input

        handle = bridge.register_forward_pre_hook(_rec)
        x = torch.randn(1, self.IN_CH, 16, 16)
        t = torch.zeros(1, dtype=torch.long)
        try:
            with torch.no_grad():
                model(x, t, smaps=self._smaps(1, 16, 16), **_measurement_kwargs(x))
        finally:
            handle.remove()
        assert seen == [2 * self.IN_CH], (
            f"backbone saw {seen} channels, expected {[2 * self.IN_CH]}"
        )

    def test_complex_and_real_maps_produce_the_same_width(self) -> None:
        """Map dtype must be aligned before the concat. ``torch.cat`` promotes a
        real stack to complex, and the bridge's entry transform then interleaves
        C complex channels into 2C real ones — so raw complex maps silently
        widened the stack instead of failing."""
        model = self._build()
        x = torch.randn(1, self.IN_CH, 16, 16)
        t = torch.zeros(1, dtype=torch.long)
        maps_c = self._smaps(1, 16, 16)
        maps_r = torch.view_as_real(maps_c).permute(0, 1, 4, 2, 3).reshape(
            1, self.IN_CH, 16, 16
        )
        seen: list[int] = []
        handle = self._record_widths(self._bridge(model), seen)
        try:
            with torch.no_grad():
                a = model(x, t, smaps=maps_c, **_measurement_kwargs(x))
                b = model(x, t, smaps=maps_r, **_measurement_kwargs(x))
        finally:
            handle.remove()
        a = a[0] if isinstance(a, tuple) else a
        b = b[0] if isinstance(b, tuple) else b
        assert a.shape == b.shape
        assert seen == [2 * self.IN_CH, 2 * self.IN_CH], (
            f"backbone saw {seen}; complex maps must be real-interleaved before "
            "the concat, or torch.cat promotes the whole stack to complex and "
            "the entry transform doubles it again."
        )

    def test_repeat_forwards_across_a_width_flip_are_bit_identical(self) -> None:
        """THE regression. Pre-fix this differential was 137% on the real
        experiment_11 model: the flip to the training width destroyed the
        adapter, and the next validation drew a fresh random one."""
        model = self._build()
        x_bare = torch.randn(1, self.IN_CH, 16, 16)
        x_train = torch.randn(1, 2 * self.IN_CH, 16, 16)
        smaps = self._smaps(1, 16, 16)
        t = torch.zeros(1, dtype=torch.long)
        mk = _measurement_kwargs(x_bare)
        with torch.no_grad():
            first = model(x_bare, t, smaps=smaps, **mk)
            model(x_train, t, **_measurement_kwargs(x_train))  # a training step
            second = model(x_bare, t, smaps=smaps, **mk)
        first = first[0] if isinstance(first, tuple) else first
        second = second[0] if isinstance(second, tuple) else second
        assert torch.equal(first, second), (
            f"two identical validations disagree by "
            f"{(first - second).abs().max().item():.6f}; the model is "
            "non-deterministic across a width flip."
        )

    def test_sample_threads_smaps_and_restores_the_stash(self) -> None:
        """``sample()`` must stash the maps for the whole reverse loop (the
        sampler re-enters ``forward(x_t, t)`` with no kwargs) and restore the
        previous value afterwards, so a validation cannot leak its maps into the
        next training step."""
        model = self._build()
        model.set_current_smaps(None)
        meas = torch.randn(1, self.IN_CH, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        mask[..., ::2] = 0
        seen: list[int] = []
        handle = self._record_widths(self._bridge(model), seen)
        try:
            with torch.no_grad():
                out = model.sample(
                    measurement=meas,
                    mask=mask,
                    inference_timesteps=2,
                    smaps=self._smaps(1, 16, 16),
                )
        finally:
            handle.remove()
        assert out.shape == meas.shape
        assert torch.isfinite(out).all()
        assert seen and set(seen) == {2 * self.IN_CH}, (
            f"reverse steps ran at widths {seen}; EVERY step must see the "
            "trained width. A stash set once and then overwritten by forward() "
            "conditions step 1 only."
        )
        assert getattr(model, "_current_smaps", None) is None, (
            "sample() leaked its S-maps into the generator's stash"
        )

    def test_sample_without_smaps_raises_rather_than_reconstructing_noise(self) -> None:
        """PLANTED VIOLATION: the pre-fix multi-step validation call verbatim."""
        model = self._build()
        meas = torch.randn(1, self.IN_CH, 16, 16)
        mask = torch.ones(1, 1, 16, 16)
        with pytest.raises(ValueError, match=r"built for"), torch.no_grad():
            model.sample(measurement=meas, mask=mask, inference_timesteps=2)


class TestDynamicMaskWiring:
    """``acceleration.enable_dynamic_mask`` was a dead façade knob (declared in
    the schema + YAMLs, read nowhere — pitfall #15). It must now flow through the
    generator into the degradation operator so the YAML knob is real: when on,
    the model trains on a fresh undersampling pattern per sample instead of one
    fixed pattern per acceleration level.
    """

    def _build(self, dynamic: bool) -> KSpaceColdDiffusionGenerator:
        return KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=16,
            num_layers=2,
            attention_type="channel",
            process_type="cold_diffusion",
            acceleration_config={
                "acceleration_type": "equispaced",
                "enable_dynamic_mask": dynamic,
            },
        )

    def test_enable_dynamic_mask_reaches_kspace_process(self) -> None:
        assert self._build(True).kspace_process.enable_dynamic_mask is True

    def test_default_dynamic_mask_is_off(self) -> None:
        assert self._build(False).kspace_process.enable_dynamic_mask is False


class TestInternalDCBackboneSmapChannels:
    """Regression: unrolled backbones with an internal ``MaskedReplacementDataConsistency``
    (``diff_varnet`` / ``diff_varnet_kan``) must NOT have their input channels
    doubled for S-map conditioning.

    Crash (cluster run ``experiment_11_kspace_cold_diffusion_varnet``,
    2026-06-14, iter 1)::

        data_consistency_layer.py:154
        RuntimeError: The size of tensor a (8) must match the size of tensor b
        (4) at non-singleton dimension 1

    Root cause: with ``condition_with_smaps=True`` (default) the generator built
    the backbone with ``in_channels * 2`` to accept the strategy's
    ``[noisy || smaps]`` concatenation. But ``diff_varnet`` applies DC against
    the **un-doubled** measured k-space at every cascade and PRESERVES the
    channel count end-to-end (no final projection to ``out_channels`` — unlike
    ``swin_diff_rec``, whose ``final_conv`` reduces to ``out_channels`` *before*
    its single DC). So ``k_guessed`` (``in_channels // 2`` complex) mismatched
    ``measured`` (``in_channels // 4`` complex) inside the DC.

    Fix: skip the S-map channel-doubling for the internal-DC backbones
    (``KSpaceColdDiffusionGenerator._INTERNAL_DC_BACKBONES``). ``complex_unet``
    (no internal DC) keeps the doubling.

    S-map conditioning does NOT "still reach them via the learned ChannelAdapter
    1x1 projection" — that adapter is constructed inside
    ``FourierBridgeNetwork.forward``, i.e. after the optimizer captured
    ``model.parameters()``, so it is never trained, never checkpointed and is
    redrawn on every width change (#1326). These backbones simply receive no
    S-maps, and the width mismatch that used to be absorbed by that adapter now
    raises. ``expects_smaps_concat`` is the one predicate that decides both the
    backbone's width and whether any caller may concatenate.
    """

    def _build(self, backbone_type: str) -> KSpaceColdDiffusionGenerator:
        return KSpaceColdDiffusionGenerator(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            num_layers=2,
            backbone_type=backbone_type,
            # force_pure_kspace is incidental to what this class asserts (S-map
            # channel arithmetic). It was True; the internal-DC backbones now
            # reject that pairing, because their MaskedReplacementDataConsistency takes an
            # IMAGE and FFTs it, so skipping the bridge's entry ifft2c applied DC
            # in the wrong domain. False is the correct pairing and leaves the
            # channel-doubling behaviour under test unchanged. complex_unet (the
            # scope-guard case below) is k-space-native and unaffected either way.
            force_pure_kspace=False,
            use_complex_conv=True,
            timesteps=28,
        )

    @pytest.mark.parametrize("backbone_type", ["diff_varnet", "diff_varnet_kan"])
    def test_internal_dc_backbone_input_channels_not_doubled(self, backbone_type: str) -> None:
        """The unrolled core is built with ``in_channels == 8`` (the measured
        k-space width), not the doubled ``16`` — otherwise its internal DC
        crashes against the un-doubled measured k-space."""
        model = self._build(backbone_type)
        core = model.backbone.backbone  # FourierBridgeNetwork -> DiffVarNet(KAN)
        assert core.in_channels == 8, (
            f"{backbone_type} built with in_channels={core.in_channels}; the "
            "S-map channel-doubling must be skipped for internal-DC backbones."
        )

    @pytest.mark.parametrize("backbone_type", ["diff_varnet", "diff_varnet_kan"])
    def test_internal_dc_backbone_does_not_expect_smaps(self, backbone_type: str) -> None:
        """``expects_smaps_concat`` is False for the internal-DC backbones, and
        it agrees with the width their backbone was actually built at.

        This is the elected owner (CLAUDE.md #17): the strategy reads it to
        decide whether to concatenate, so the two can no longer disagree."""
        model = self._build(backbone_type)
        assert model.expects_smaps_concat is False
        assert model.backbone.config.in_channels == model.in_channels

    def test_doubled_input_into_internal_dc_backbone_raises(self) -> None:
        """A ``[noisy || smaps]`` (2x ``in_channels``) stack handed to a backbone
        built at ``in_channels`` must RAISE.

        It used to be silently absorbed by a rebuilt ChannelAdapter, so every
        diff_varnet arm trained through an untrained 1x1 projection for its whole
        run. The strategy no longer produces this stack for these backbones; a
        caller that still does is now told."""
        model = self._build("diff_varnet").eval()
        x_doubled = torch.randn(1, 16, 32, 32)
        t = torch.zeros(1, dtype=torch.long)
        with pytest.raises(ValueError, match="channel"), torch.no_grad():
            model(x_doubled, t, **_measurement_kwargs(x_doubled))

    def test_non_internal_dc_backbone_still_doubled_for_smaps(self) -> None:
        """Scope guard: ``complex_unet`` has NO internal DC, so it keeps the
        doubled (2x ``in_channels``) input for S-map conditioning — the fix must
        not over-correct working backbones."""
        model = self._build("complex_unet")
        # FourierBridgeNetwork.config.in_channels carries the build-time width.
        assert model.backbone.config.in_channels == 16


class TestWaveletGateDispatch:
    """2026-07-01: ``gate_type`` must flow through the KSpaceDownsample/Upsample
    dispatch to ``WaveletFreqAttentionBlock``. Previously the ``wavelet_freq``
    kwarg filter dropped it, so attn_kan_wavelet == attn_mlp_wavelet (facade)."""

    def test_gate_type_reaches_wavelet_block(self):
        from spectramr.models.blocks.dual_domain_attention_kan import (
            WaveletFreqAttentionBlock,
        )
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceDownsampleBlock,
        )

        block = KSpaceDownsampleBlock(
            in_channels=8,
            out_channels=8,
            attention_type="wavelet_freq",
            kan_dual_domain_kwargs={"gate_type": "mlp", "num_levels": 1},
            feature_domain="kspace",
        )
        # `.inner`: the dispatch wraps every block in IdentityAtInitAttention
        # (issue #471). Unwrapping is deliberate -- the wrapper does NOT forward
        # attribute lookups, so a stale path raises instead of resolving silently.
        assert isinstance(block.attention.inner, WaveletFreqAttentionBlock)
        assert block.attention.inner.gate_type == "mlp"

    def test_default_gate_type_is_kan(self):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceDownsampleBlock,
        )

        block = KSpaceDownsampleBlock(
            in_channels=8,
            out_channels=8,
            attention_type="wavelet_freq",
            kan_dual_domain_kwargs={"num_levels": 1},
            feature_domain="kspace",
        )
        assert block.attention.inner.gate_type == "kan"


class TestBlockActivationValidation:
    """Pitfall #9: the k-space blocks' activation resolution must raise on an
    unknown name and on odd channel counts for the complex activations, never
    silently degrade to ReLU (the old fallback turned a mistyped/odd-channel
    'complex' request into a phase-destroying real ReLU)."""

    VALID_REAL = ("relu", "leaky_relu", "gelu")
    VALID_COMPLEX = ("complex", "modrelu")

    def test_unet_block_valid_activations_forward_finite(self):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceUNetBlock,
        )

        x = torch.randn(1, 8, 16, 16)
        for name in (*self.VALID_REAL, *self.VALID_COMPLEX):
            block = KSpaceUNetBlock(in_channels=8, out_channels=8, activation=name)
            out = block(x)
            assert out.shape == x.shape
            assert torch.isfinite(out).all()

    def test_unet_block_unknown_activation_raises(self):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceUNetBlock,
        )

        with pytest.raises(ValueError, match=r"relu.*leaky_relu.*gelu.*complex.*modrelu"):
            KSpaceUNetBlock(in_channels=8, out_channels=8, activation="silu")

    def test_unet_block_odd_channels_complex_raises(self):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceUNetBlock,
        )

        with pytest.raises(ValueError, match="even"):
            KSpaceUNetBlock(in_channels=8, out_channels=3, activation="complex")

    def test_downsample_block_valid_activations_forward_finite(self):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceDownsampleBlock,
        )

        x = torch.randn(1, 8, 16, 16)
        for name in (*self.VALID_REAL, *self.VALID_COMPLEX):
            block = KSpaceDownsampleBlock(
                in_channels=8,
                out_channels=8,
                attention_type="none",
                activation=name,
                feature_domain="kspace",
            )
            out, skip = block(x)
            assert torch.isfinite(out).all()
            assert torch.isfinite(skip).all()

    def test_downsample_block_unknown_activation_raises(self):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceDownsampleBlock,
        )

        with pytest.raises(ValueError, match=r"relu.*leaky_relu.*gelu.*complex.*modrelu"):
            KSpaceDownsampleBlock(
                in_channels=8,
                out_channels=8,
                attention_type="none",
                activation="swish",
                feature_domain="kspace",
            )

    @pytest.mark.parametrize("name", VALID_COMPLEX)
    def test_downsample_block_odd_channels_complex_raises(self, name):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceDownsampleBlock,
        )

        with pytest.raises(ValueError, match="even"):
            KSpaceDownsampleBlock(
                in_channels=8,
                out_channels=3,
                attention_type="none",
                activation=name,
                feature_domain="kspace",
            )

    def test_upsample_block_valid_activations_forward_finite(self):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceUpsampleBlock,
        )

        x = torch.randn(1, 8, 8, 8)
        skip = torch.randn(1, 8, 16, 16)
        for name in (*self.VALID_REAL, *self.VALID_COMPLEX):
            block = KSpaceUpsampleBlock(
                in_channels=8,
                out_channels=8,
                attention_type="none",
                activation=name,
                feature_domain="kspace",
            )
            out = block(x, skip)
            assert torch.isfinite(out).all()

    def test_upsample_block_unknown_activation_raises(self):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceUpsampleBlock,
        )

        with pytest.raises(ValueError, match=r"relu.*leaky_relu.*gelu.*complex.*modrelu"):
            KSpaceUpsampleBlock(
                in_channels=8,
                out_channels=8,
                attention_type="none",
                activation="swish",
                feature_domain="kspace",
            )

    @pytest.mark.parametrize("name", VALID_COMPLEX)
    def test_upsample_block_odd_channels_complex_raises(self, name):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceUpsampleBlock,
        )

        with pytest.raises(ValueError, match="even"):
            KSpaceUpsampleBlock(
                in_channels=8,
                out_channels=3,
                attention_type="none",
                activation=name,
                feature_domain="kspace",
            )

    def test_generator_default_complex_activation_still_builds(self):
        """The generator ships activation='complex' by default — the tightened
        validation must not reject the shipped default.

        attention_type='channel' sidesteps a pre-existing, unrelated crash:
        the generator default 'self' is rejected by ConfigurableResidualBlock
        in reconstruction/unet.py (fails identically at HEAD)."""
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            attention_type="channel",
        )
        assert isinstance(model, nn.Module)


class TestFeatureDomainDerivation:
    """force_pure_kspace must derive the feature_domain threaded into every
    domain-aware sub-block of the complex_unet backbone (2026-07-03)."""

    _AWARE = (
        "DualDomainBlock",
        "DualDomainAttention",
        "KANGatedDualDomainAttention",
        "WaveletFreqAttentionBlock",
    )

    def _domains(self, model):
        return {
            getattr(m, "feature_domain", None)
            for m in model.modules()
            if type(m).__name__ in self._AWARE
        }

    def test_force_pure_kspace_derives_kspace_domain(self):
        model = KSpaceColdDiffusionGenerator(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            num_res_blocks=2,
            attention_type="kan_dual_domain",
            backbone_type="complex_unet",
            force_pure_kspace=True,
            use_complex_conv=True,
            activation="complex",
            time_embedding_dim=64,
            timesteps=28,
            kan_dual_domain_kwargs={"num_heads": 2, "num_bands": 4, "kan_hidden": 8},
        )
        assert self._domains(model) == {"kspace"}

    def test_bridge_mode_derives_image_domain(self):
        model = KSpaceColdDiffusionGenerator(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            num_res_blocks=2,
            attention_type="dual_domain",
            backbone_type="complex_unet",
            force_pure_kspace=False,
            use_complex_conv=True,
            activation="complex",
            time_embedding_dim=64,
            timesteps=28,
        )
        assert self._domains(model) == {"image"}

    def test_pure_kspace_unet_rejects_nonnull_attention(self):
        """backbone=unet + force_pure_kspace builds PureKSpaceUNet (no attention
        seam) — a non-'none' attention_type must fail loud, not be dropped."""
        with pytest.raises(ValueError, match="PureKSpaceUNet"):
            KSpaceColdDiffusionGenerator(
                in_channels=8,
                out_channels=8,
                base_channels=16,
                num_res_blocks=2,
                attention_type="self",
                backbone_type="unet",
                force_pure_kspace=True,
                use_complex_conv=True,
                activation="complex",
                time_embedding_dim=64,
                timesteps=28,
            )

    def test_pure_kspace_unet_builds_with_none_attention(self):
        model = KSpaceColdDiffusionGenerator(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            num_res_blocks=2,
            attention_type="none",
            backbone_type="unet",
            force_pure_kspace=True,
            use_complex_conv=True,
            activation="complex",
            time_embedding_dim=64,
            timesteps=28,
        )
        assert isinstance(model, nn.Module)


class TestReverseSamplingModeKnob:
    """The ``reverse_sampling_mode`` / ``reverse_clip_ratio`` model_kwargs knobs
    are read, validated at BUILD (pitfall #15), and stamped on the generator."""

    def test_unknown_reverse_sampling_mode_raises_at_build(self):
        """An illegal value fails at construction, not mid-validation."""
        with pytest.raises(ValueError, match="reverse_sampling_mode"):
            KSpaceColdDiffusionGenerator(
                in_channels=2,
                out_channels=2,
                base_channels=8,
                num_layers=2,
                reverse_sampling_mode="bogus",
            )

    def test_nonpositive_reverse_clip_ratio_raises_at_build(self):
        with pytest.raises(ValueError, match="reverse_clip_ratio"):
            KSpaceColdDiffusionGenerator(
                in_channels=2,
                out_channels=2,
                base_channels=8,
                num_layers=2,
                reverse_clip_ratio=-1.0,
            )

    def test_default_is_additive(self):
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            attention_type="none",
        )
        assert model._reverse_mode == "additive"
        assert model._reverse_clip_ratio == pytest.approx(4.0)

    def test_replace_freeze_stamped(self):
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            attention_type="none",
            reverse_sampling_mode="replace_freeze",
            reverse_clip_ratio=3.0,
        )
        assert model._reverse_mode == "replace_freeze"
        assert model._reverse_clip_ratio == pytest.approx(3.0)


class TestKSpaceFeatureNormPlumbing:
    """``kspace_feature_norm`` reaches the complex_unet backbone; misuse raises."""

    def test_rms_reaches_complex_unet_backbone(self):
        """The model_kwarg threads generator -> FourierBridgeNetwork -> ComplexUNet
        and materializes ComplexRMSNorm modules (the experiment_11 divergence fix)."""
        from spectramr.models.layers.complex_norm import ComplexRMSNorm

        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            use_complex_conv=True,
            attention_type="none",
            force_pure_kspace=True,
            backbone_type="complex_unet",
            kspace_feature_norm="rms",
        )
        norms = [m for m in model.modules() if isinstance(m, ComplexRMSNorm)]
        assert len(norms) >= 1

    def test_misapplied_to_non_complex_unet_raises(self):
        """Setting the knob on a backbone that cannot consume it must fail loud
        (pitfall #15: no silently-swallowed knob), not no-op."""
        with pytest.raises(ValueError, match="complex_unet"):
            KSpaceColdDiffusionGenerator(
                in_channels=2,
                out_channels=2,
                base_channels=8,
                num_layers=2,
                use_complex_conv=False,
                attention_type="none",
                backbone_type="unet",
                kspace_feature_norm="rms",
            )

    def test_the_pure_kspace_branch_is_gated_too(self):
        """The branch the gate used to miss entirely.

        ``force_pure_kspace`` + ``backbone_type: 'unet'`` builds PureKSpaceUNet
        **directly** and never constructs FourierBridgeNetwork, where the gate
        used to live -- so this arm built without a word and materialised zero
        ``ComplexRMSNorm`` modules while declaring the knob. Planted here because
        no arm in the corpus exercises the combination, which is why it survived.
        """
        with pytest.raises(ValueError, match="complex_unet"):
            KSpaceColdDiffusionGenerator(
                in_channels=2,
                out_channels=2,
                base_channels=8,
                num_layers=2,
                use_complex_conv=True,
                attention_type="none",
                force_pure_kspace=True,
                backbone_type="unet",
                kspace_feature_norm="rms",
            )

    def test_the_declared_time_embedding_width_reaches_the_backbone(self):
        """It was hardcoded to 256 at the bridge, so a declared value was inert."""
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            use_complex_conv=True,
            attention_type="none",
            force_pure_kspace=True,
            backbone_type="complex_unet",
            time_embedding_dim=64,
        )
        assert model.backbone.backbone.time_embedding_dim == 64


def test_both_block_dispatches_wrap_attention_identity_at_init() -> None:
    """Down AND up blocks must wrap (issue #471).

    The up-block dispatch is a near-copy of the down-block one, so a fix applied to
    only one of them is the easy mistake -- and it is exactly what happened with
    ``spatial``, which exists in the up dispatch alone.
    """
    from spectramr.models.blocks.attention import IdentityAtInitAttention
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceDownsampleBlock,
        KSpaceUpsampleBlock,
    )

    down = KSpaceDownsampleBlock(
        in_channels=16,
        out_channels=16,
        use_complex_conv=True,
        activation="modrelu",
        time_embedding_dim=64,
        attention_type="channel",
        feature_domain="kspace",
    )
    up = KSpaceUpsampleBlock(
        in_channels=16,
        out_channels=16,
        use_complex_conv=True,
        activation="modrelu",
        time_embedding_dim=64,
        attention_type="channel",
        feature_domain="kspace",
    )

    for blk in (down, up):
        assert isinstance(blk.attention, IdentityAtInitAttention)
        assert blk.attention.gamma.item() == 0.0


def test_attention_type_none_is_not_wrapped() -> None:
    """The control must stay a bare Identity: energy_probe skips on isinstance."""
    from torch import nn

    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceDownsampleBlock,
    )

    blk = KSpaceDownsampleBlock(
        in_channels=16,
        out_channels=16,
        use_complex_conv=True,
        activation="modrelu",
        time_embedding_dim=64,
        attention_type="none",
        feature_domain="kspace",
    )
    assert isinstance(blk.attention, nn.Identity)


class TestAccelerationConfigObjectContract:
    """``acceleration_config`` arrives in three different shapes (issue #550).

    ``ModelBuilder`` injects the live ``AccelerationConfigSchema``,
    ``ModelFactory`` injects its ``model_dump()``, and tests/scripts pass raw
    dicts. The constructor annotates the parameter ``dict | None`` and used to
    call ``.get`` on it directly, so only the factory path worked: any caller
    that built the generator from a config object hit an AttributeError. All
    three must produce the same degradation operator.
    """

    ACCEL: ClassVar[dict] = {
        "acceleration_type": "equispaced",
        "schedule_type": "step",
        "base_acceleration": 2.0,
        "max_acceleration": 32.0,
        "center_fraction": 0.08,
        "min_center_fraction": 0.02,
        "acceleration_range": [2.0, 4.0, 8.0, 10.0, 12.0, 16.0, 32.0],
    }

    def _build(self, acceleration_config) -> KSpaceColdDiffusionGenerator:
        return KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=16,
            num_layers=2,
            attention_type="channel",
            process_type="cold_diffusion",
            timesteps=28,
            acceleration_config=acceleration_config,
        )

    def _fingerprint(self, model: KSpaceColdDiffusionGenerator) -> tuple:
        process = model.kspace_process
        return (
            process.center_fraction,
            process.min_center_fraction,
            process.max_accel,
            process.base_acceleration,
            process.mask_type,
            tuple(e for _t, _n, e, _k in process.describe_ladder((256, 256))),
        )

    def test_schema_object_matches_raw_dict(self) -> None:
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema

        config = AccelerationConfigSchema(**self.ACCEL)
        assert self._fingerprint(self._build(config)) == self._fingerprint(self._build(self.ACCEL))

    def test_model_dump_matches_raw_dict(self) -> None:
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema

        dumped = AccelerationConfigSchema(**self.ACCEL).model_dump()
        assert self._fingerprint(self._build(dumped)) == self._fingerprint(self._build(self.ACCEL))

    def test_min_center_fraction_reaches_the_degradation_operator(self) -> None:
        from spectramr.config.schemas.acceleration import AccelerationConfigSchema

        model = self._build(AccelerationConfigSchema(**self.ACCEL))
        assert model.kspace_process.min_center_fraction == 0.02
        assert model.kspace_process.declared_ladder_defects((256, 256)) == []


# ---------------------------------------------------------------------------
# The default build must be constructible (regression, cluster job 8004252)
# ---------------------------------------------------------------------------


def _tiny(**kw):
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceColdDiffusionGenerator,
    )

    return KSpaceColdDiffusionGenerator(
        in_channels=2,
        out_channels=2,
        base_channels=8,
        num_layers=2,
        use_complex_conv=False,
        timesteps=28,
        **kw,
    )


class TestAttentionDefaultIsBackboneAware:
    """``backbone_type`` and ``attention_type`` defaulted to an INVALID pair.

    ``backbone_type`` defaults to ``'unet'`` and ``attention_type`` defaulted to
    ``'self'``, which the unet branch rejects — so the zero-extra-kwargs build
    raised, and **22 corpus arms declaring neither knob** could not construct
    their model at all. Four of cluster job 8004252's failures were tests that
    pass neither knob and were reported as configuration errors.

    The guard itself is right and stays: an explicit unsupported request must
    fail loud (pitfall #9). What changed is that it can no longer fire on a
    value nobody asked for — "a new validation must not reject the library
    default; guard the explicit argument, not the resolved one."
    """

    def test_default_build_constructs(self) -> None:
        assert _tiny() is not None

    def test_unet_default_resolves_to_the_blocks_own_default(self) -> None:
        """``none`` is ConfigurableResidualBlock's own default, not a third answer.

        Pinning the VALUE, not just "it constructs": resolving to ``spatial`` or
        ``channel`` would also construct while silently giving the arm an
        attention mechanism its author never asked for.
        """
        import inspect

        from spectramr.models.reconstruction.unet import ConfigurableResidualBlock

        block_default = (
            inspect.signature(ConfigurableResidualBlock.__init__)
            .parameters["attention_type"]
            .default
        )
        assert block_default.value == "none"

    def test_explicit_unsupported_attention_still_raises(self) -> None:
        """The guard is intact for a value the user actually asked for."""
        with pytest.raises(ValueError, match="backbone_type='unet'"):
            _tiny(attention_type="self")

    def test_explicit_supported_attention_is_accepted(self) -> None:
        assert _tiny(attention_type="spatial") is not None

    def test_complex_unet_keeps_self_attention(self) -> None:
        """The other backbone's default is unchanged — this is not a global swap."""
        assert _tiny(backbone_type="complex_unet") is not None


# ---------------------------------------------------------------------------
# Seamless-backbone attention guard
#
# swin_diff_rec / diff_varnet / nafnet (and the two KAN twins) contain zero
# references to ``attention_type`` and absorb it via ``**kwargs``. Before this
# guard the value was validated against the registry and then discarded, so an
# arm could advertise attention it never ran (pitfall #16 facade) — measured on
# swin_diff_rec, every legal value produced a byte-identical 26.40M-parameter
# model while a bogus value still raised.
#
# The five restoration/ViT backbones joined the set on 2026-09-17, on the sweep
# the guard's comment had been waiting for: every registered attention_type
# leaves each of their parameter counts exactly unchanged, and ``dual_domain``
# -- the only one that builds extra submodules -- adds eleven carrying zero
# parameters between them. vision_mamba is still absent because it cannot be
# constructed without the ``.[mamba]`` extra, so it was never measured.
# ---------------------------------------------------------------------------

_SEAMLESS = [
    "swin_diff_rec",
    "swin_diff_rec_kan",
    "diff_varnet",
    "diff_varnet_kan",
    "nafnet",
    "restormer",
    "swinir",
    "vision_transformer",
    "vit",
    "swin_transformer",
    "vision_mamba",
    "mamba_unet",
    "dit",
    "uvit",
    "u_vit",
    "hat",
    "diffit",
]

#: Names in ``_SEAMLESS`` that cannot be CONSTRUCTED in this environment, so the
#: build-side cases below skip them while the raise-side cases still run: the
#: mamba pair needs the ``.[mamba]`` extra (deliberately outside ``[dev]``).
_SEAMLESS_UNBUILDABLE = {"vision_mamba", "mamba_unet"}


@pytest.mark.parametrize("backbone", _SEAMLESS)
@pytest.mark.parametrize("attention", ["self", "channel", "dual_domain", "kan_dual_domain"])
def test_seamless_backbone_rejects_non_none_attention(backbone, attention):
    """A backbone with no attention seam must RAISE rather than silently drop
    the request. The message has to name 'none' (the fix) and 'complex_unet'
    (where the seam actually lives), so the user can act without grepping."""
    with pytest.raises(ValueError, match="no attention seam"):
        KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            backbone_type=backbone,
            attention_type=attention,
        )


@pytest.mark.parametrize("backbone", _SEAMLESS)
def test_seamless_backbone_accepts_none_attention(backbone):
    """``attention_type='none'`` is the honest declaration and must still build.
    Guarding must not make these backbones unreachable."""
    if backbone in _SEAMLESS_UNBUILDABLE:
        pytest.skip(f"{backbone} needs the .[mamba] extra to construct")
    model = KSpaceColdDiffusionGenerator(
        in_channels=2,
        out_channels=2,
        base_channels=8,
        num_layers=2,
        backbone_type=backbone,
        attention_type="none",
    )
    assert isinstance(model, nn.Module)


def test_seamless_set_excludes_the_backbone_that_has_a_seam():
    """complex_unet DOES implement the block-attention seam, so it must never be
    caught by this guard — that would delete the cohort's whole attention axis.
    Anchors the set against the class attribute rather than a literal copy."""
    assert "complex_unet" not in KSpaceColdDiffusionGenerator._SEAMLESS_ATTENTION_BACKBONES
    assert "unet" not in KSpaceColdDiffusionGenerator._SEAMLESS_ATTENTION_BACKBONES
    # unet has its own, narrower guard (none/channel/spatial) — keep them distinct.
    assert set(_SEAMLESS) == set(KSpaceColdDiffusionGenerator._SEAMLESS_ATTENTION_BACKBONES)


@pytest.mark.parametrize("backbone", _SEAMLESS)
def test_seamless_backbone_unspecified_attention_resolves_to_none(backbone):
    """ "Nobody asked" must not trip the guard. ``attention_type`` defaults to the
    None sentinel, and for a seamless backbone that has to resolve to 'none' --
    otherwise the library default ('self') would make every config that simply
    omits the knob unconstructible, which is the exact regression the unet
    branch of this resolution was written to prevent."""
    if backbone in _SEAMLESS_UNBUILDABLE:
        pytest.skip(f"{backbone} needs the .[mamba] extra to construct")
    model = KSpaceColdDiffusionGenerator(
        in_channels=2,
        out_channels=2,
        base_channels=8,
        num_layers=2,
        backbone_type=backbone,
    )
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize("backbone", ["diff_varnet", "diff_varnet_kan"])
def test_internal_dc_backbone_rejects_force_pure_kspace(backbone):
    """DiffVarNet's MaskedReplacementDataConsistency takes an IMAGE and FFTs it internally.
    ``force_pure_kspace=true`` makes FourierBridgeNetwork skip the entry ifft2c,
    so the backbone would receive k-space and DC would transform it a second
    time — data consistency enforced in the wrong domain, silently. Verified
    live that the DC layers do fire (forward-hook count), so this is a live path.
    """
    with pytest.raises(ValueError, match="force_pure_kspace"):
        KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            backbone_type=backbone,
            attention_type="none",
            force_pure_kspace=True,
        )


@pytest.mark.parametrize("backbone", ["diff_varnet", "diff_varnet_kan"])
def test_internal_dc_backbone_builds_in_bridge_mode(backbone):
    """``force_pure_kspace=false`` is the correct pairing — the bridge does the
    ifft2c and the backbone gets the image-domain tensor its DC assumes."""
    model = KSpaceColdDiffusionGenerator(
        in_channels=2,
        out_channels=2,
        base_channels=8,
        num_layers=2,
        backbone_type=backbone,
        attention_type="none",
        force_pure_kspace=False,
    )
    assert isinstance(model, nn.Module)


class TestGradCheckpointingDelegation:
    """The builder probes the GENERATOR, but the blocks live two levels down.

    ``ModelBuilder`` takes its native branch on
    ``hasattr(generator, "set_grad_checkpointing")``. The parameters are in
    ``generator.backbone.backbone`` (FourierBridgeNetwork -> ComplexUNet), so the
    call has to be forwarded or the flag is a no-op that still logs success.
    """

    @staticmethod
    def _model():
        return KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=16,
            num_layers=2,
            attention_type="none",
            backbone_type="complex_unet",
        )

    def test_model_builder_native_branch_is_reachable(self):
        """This attribute is the whole trigger for the non-generic path."""
        assert hasattr(self._model(), "set_grad_checkpointing")

    def test_enabling_reaches_the_complex_unet(self):
        model = self._model()
        unet = model.backbone.backbone
        assert hasattr(unet, "grad_checkpointing"), (
            "backbone.backbone is not the ComplexUNet — this test's target moved"
        )
        assert unet.grad_checkpointing is False

        model.set_grad_checkpointing(True)
        assert unet.grad_checkpointing is True

        model.set_grad_checkpointing(False)
        assert unet.grad_checkpointing is False

    def test_unsupported_backbone_raises_rather_than_no_ops(self):
        """Non-negotiable 3: a memory claim that cannot be honored must fail loud.

        Degrading here is what produced the original defect — an arm requesting
        checkpointing, receiving ~3% coverage, and OOMing with no clue why.
        """
        model = self._model()
        model.backbone.backbone = nn.Identity()  # a backbone with no such hook

        with pytest.raises(NotImplementedError, match="set_grad_checkpointing"):
            model.set_grad_checkpointing(True)

    def test_the_error_names_the_backbone_and_the_way_out(self):
        model = self._model()
        model.backbone.backbone = nn.Identity()

        with pytest.raises(NotImplementedError) as excinfo:
            model.set_grad_checkpointing(True)

        message = str(excinfo.value)
        assert "complex_unet" in message, "does not point at a backbone that works"
        assert "enable_checkpointing" in message, "does not name the knob to unset"


# ---------------------------------------------------------------------------
# The generator's own fallback concat also FFTs the maps first (#1297)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fallback_concat_prepares_smaps_and_leaves_sense_input_untouched() -> None:
    """The in-generator concat is a fourth stack-building site.

    It only fires for backbones outside ``_no_concat_backbones`` (so not for
    exp11's ``complex_unet``), but when it does it must match the strategy or
    those arms train on a different stack than the rest. Critically the raw
    ``smaps`` name must survive: ``self.sense_projector(x_out, smaps)`` further
    down is the ONE physically-correct use of the maps in the forward pass, and
    it needs them in image space.
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(KSpaceColdDiffusionGenerator.forward))
    tree = ast.parse(src)

    prepared = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "prepare_smaps_for_kspace_conditioning"
    ]
    assert len(prepared) == 1

    for node in ast.walk(tree):
        is_cat = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "cat"
            and node.args
            and isinstance(node.args[0], (ast.List, ast.Tuple))
        )
        if not is_cat:
            continue
        names = [e.id for e in node.args[0].elts if isinstance(e, ast.Name)]
        if "smaps_k" in names or "smaps" in names:
            assert "smaps" not in names, "raw image-domain maps concatenated"

    # the SENSE projection still receives the untransformed maps
    assert "self.sense_projector(x_out, smaps)" in src


class TestTheOneWidthResolver:
    """``model_expects_smaps_concat`` is the single owner of "is the stack
    doubled?" (CLAUDE.md #17).

    Four call sites used to answer it independently -- the training strategy,
    ``forward_probe``, ``energy_probe`` and ``ColdDiffusionInferenceStrategy``
    -- three by reading ``condition_with_smaps`` and the fourth
    (``_assert_trained_width``) by hard-coding ``2 * in_channels``. All four
    were wrong for the internal-DC backbones, whose declaration is honoured and
    whose backbone is still built at 1x.
    """

    @staticmethod
    def _gen(backbone_type: str):
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceColdDiffusionGenerator,
        )

        kwargs = {
            "in_channels": 8,
            "out_channels": 8,
            "backbone_type": backbone_type,
            "condition_with_smaps": True,
            "base_channels": 8,
            "num_timesteps": 4,
        }
        if backbone_type.startswith("diff_varnet"):
            # diff_varnet's own DC is incompatible with the pure-k-space path.
            kwargs["force_pure_kspace"] = False
        return KSpaceColdDiffusionGenerator(**kwargs)

    @staticmethod
    def _first_conv_width(model) -> int:
        """What the backbone actually accepts -- read off the built module.

        Asserting against the attribute would be circular; this reads the real
        first weighted layer, so a wrong answer is a wrong *number*.
        """
        for _name, mod in model.named_modules():
            width = getattr(mod, "in_channels", None)
            if isinstance(width, int) and getattr(mod, "weight", None) is not None:
                return width
        raise AssertionError("no weighted layer with in_channels found")

    @pytest.mark.parametrize(
        ("backbone_type", "doubled"),
        [
            ("complex_unet", True),
            ("diff_varnet", False),
            ("diff_varnet_kan", False),
        ],
    )
    def test_the_resolver_matches_the_real_backbone_width(
        self, backbone_type: str, doubled: bool
    ) -> None:
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            model_expects_smaps_concat,
        )

        model = self._gen(backbone_type)
        # Every arm here DECLARES conditioning; only the resolved answer differs.
        assert model.condition_with_smaps is True
        assert model_expects_smaps_concat(model) is doubled
        expected = 16 if doubled else 8
        assert self._first_conv_width(model) == expected

    def test_the_resolved_contract_outranks_the_declaration(self) -> None:
        """A stub that declares conditioning and disowns the concat resolves to
        the concat's answer -- this is the divergence every blind site missed."""
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            model_expects_smaps_concat,
        )

        internal_dc = SimpleNamespace(
            condition_with_smaps=True, expects_smaps_concat=False
        )
        assert model_expects_smaps_concat(internal_dc) is False
        assert model_expects_smaps_concat(internal_dc, default=True) is False

    def test_a_pre_contract_model_falls_back_to_its_declaration(self) -> None:
        """Models predating ``expects_smaps_concat`` keep their old answer; this
        is the one place in the codebase that substitution is allowed."""
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            model_expects_smaps_concat,
        )

        assert model_expects_smaps_concat(SimpleNamespace(condition_with_smaps=True))
        assert not model_expects_smaps_concat(
            SimpleNamespace(condition_with_smaps=False)
        )
        # ...and the declaration still outranks the caller's default, because it
        # is a real statement about the model rather than an absence.
        assert not model_expects_smaps_concat(
            SimpleNamespace(condition_with_smaps=False), default=True
        )

    def test_the_default_differs_by_call_site_and_must_be_explicit(self) -> None:
        """A model carrying neither attribute has no answer of its own.

        The probes and the inference path read ``getattr(..., False)`` before
        this resolver existed and must keep answering False; the training
        strategy concatenated unconditionally and must keep answering True.
        Collapsing the two would silently change one of them.
        """
        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            model_expects_smaps_concat,
        )

        bare = SimpleNamespace()
        assert model_expects_smaps_concat(bare) is False
        assert model_expects_smaps_concat(bare, default=True) is True


class TestSampleStartTimestepForwarding:
    """``sample()`` hands the trajectory head to samplers that understand it.

    ``start_timestep`` is a cold_mri concept (#535/#1388): it starts the reverse
    trajectory at the timestep the measurement is actually degraded at instead of
    replaying the fully-degraded schedule. The posterior samplers
    (dps_posterior / pi_gdm / dds / red / pnp_admm) route through this SAME method
    and their ``sample()`` does not take it, so it is forwarded by signature
    inspection. The failure mode being pinned is the quiet one: dropping a value
    the caller explicitly set (pitfall #15).
    """

    @staticmethod
    def _generator() -> KSpaceColdDiffusionGenerator:
        model = KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            attention_type="none",
        )
        # Any non-cold name: keeps the cold_mri-only kwargs out of the way so the
        # test exercises forwarding rather than sampler construction.
        model._sampler_name = "stub_sampler"
        return model

    @staticmethod
    def _patch(monkeypatch, sampler) -> None:
        import spectramr.models.diffusion.samplers as samplers

        monkeypatch.setattr(samplers, "get_sampler", lambda name, **kw: sampler)

    @staticmethod
    def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
        measurement = torch.randn(1, 2, 16, 16)
        mask = torch.zeros(1, 1, 16, 16)
        mask[..., ::2] = 1.0
        return measurement * mask, mask

    def test_forwarded_when_the_sampler_accepts_it(self, monkeypatch) -> None:
        """A cold_mri-style sampler receives the head verbatim."""

        class Accepting:
            received: ClassVar[list] = []

            def sample(self, measurement, mask, start_timestep=None):
                Accepting.received.append(start_timestep)
                return measurement

        Accepting.received.clear()
        self._patch(monkeypatch, Accepting())
        measurement, mask = self._inputs()

        self._generator().sample(measurement, mask=mask, start_timestep=7)

        assert Accepting.received == [7]

    def test_absent_when_the_caller_did_not_ask(self, monkeypatch) -> None:
        """``None`` must not be forwarded as an explicit ``start_timestep=None``.

        The legacy call has to stay byte-for-byte identical for every sampler,
        including ones whose ``sample()`` takes no such parameter at all.
        """

        class Strict:
            seen_kwargs: ClassVar[list] = []

            def sample(self, measurement, mask, **kwargs):
                Strict.seen_kwargs.append(dict(kwargs))
                return measurement

        Strict.seen_kwargs.clear()
        self._patch(monkeypatch, Strict())
        measurement, mask = self._inputs()

        self._generator().sample(measurement, mask=mask)

        assert Strict.seen_kwargs == [{}]

    def test_unsupported_sampler_warns_instead_of_dropping(
        self, monkeypatch, caplog
    ) -> None:
        """The posterior samplers cannot honour it -- so say so, out loud.

        Silently ignoring it would leave a caller believing the trajectory was
        shortened while it still ran from the fully-degraded head.
        """
        import logging

        class Unsupported:
            def sample(self, measurement, mask):
                return measurement

        self._patch(monkeypatch, Unsupported())
        measurement, mask = self._inputs()

        with caplog.at_level(
            logging.WARNING,
            logger="spectramr.models.generators.kspace_cold_diffusion_generator",
        ):
            self._generator().sample(measurement, mask=mask, start_timestep=7)

        # ``getMessage()`` -- ``.message`` is only populated once a handler has
        # formatted the record, which is order-dependent across a wide run.
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any("start_timestep" in m for m in warnings), warnings

    def test_unsupported_sampler_still_reconstructs(self, monkeypatch) -> None:
        """Warning, not raising: an unsupported sampler must still return a result."""

        class Unsupported:
            def sample(self, measurement, mask):
                return measurement

        self._patch(monkeypatch, Unsupported())
        measurement, mask = self._inputs()

        out = self._generator().sample(measurement, mask=mask, start_timestep=7)

        assert out.shape == measurement.shape


class TestForcePureKSpaceDomainGuard:
    """``force_pure_kspace`` must be rejected for every backbone whose internal
    DC is image-domain -- not just the two in the channel-width set.

    ``FourierBridgeNetwork`` skips the entry ``ifft2c`` under the flag, so the
    inner backbone receives k-space. A backbone that then runs its own
    ``MaskedReplacementDataConsistency`` FFTs that k-space a SECOND time and enforces data
    consistency in the wrong domain. There is no shape error, so nothing
    surfaces at runtime -- the arm trains and reports metrics.

    The guard used to key on ``_INTERNAL_DC_BACKBONES``, whose membership
    answers the unrelated CHANNEL-WIDTH question, so ``swin_diff_rec`` and
    ``swin_diff_rec_kan`` -- image-domain DC, zero ``fft2c``/``torch.fft`` in
    either module -- went unchecked (CLAUDE.md #17: one owner per invariant).

    These construct nothing: the guard fires while only a ``UNetConfig``
    dataclass exists, before ``FourierBridgeNetwork`` is built.
    """

    @staticmethod
    def _kwargs(backbone_type: str) -> dict:
        return dict(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            num_res_blocks=2,
            attention_type="none",
            backbone_type=backbone_type,
            force_pure_kspace=True,
            use_complex_conv=True,
            activation="complex",
            time_embedding_dim=64,
            timesteps=28,
        )

    # One case per SHAPE the rule takes (CLAUDE.md #15): the newly-covered swin
    # backbone, its KAN sibling, and the varnet case that was already covered
    # and must not regress.
    @pytest.mark.parametrize(
        "backbone_type",
        ["swin_diff_rec", "swin_diff_rec_kan", "diff_varnet", "diff_varnet_kan"],
    )
    def test_image_domain_dc_backbone_rejects_force_pure_kspace(
        self, backbone_type: str
    ) -> None:
        with pytest.raises(ValueError, match="wrong domain"):
            KSpaceColdDiffusionGenerator(**self._kwargs(backbone_type))

    def test_domain_set_is_a_strict_superset_of_the_width_set(self) -> None:
        """The two sets answer different questions and must not be collapsed.

        If a later change makes them equal, one of the two invariants has
        silently lost its owner -- which is the state this fix repaired.
        """
        width = KSpaceColdDiffusionGenerator._INTERNAL_DC_BACKBONES
        domain = KSpaceColdDiffusionGenerator._IMAGE_DOMAIN_DC_BACKBONES
        assert width < domain
        assert {"swin_diff_rec", "swin_diff_rec_kan"} <= domain - width

    def test_kspace_native_backbone_stays_out_of_the_domain_set(self) -> None:
        """``complex_unet`` has no internal DC, so force_pure_kspace is a
        legitimate configuration for it and must not be swept up."""
        assert (
            "complex_unet"
            not in KSpaceColdDiffusionGenerator._IMAGE_DOMAIN_DC_BACKBONES
        )


class TestRepetitionFusionCapabilityProbes:
    """The two capability probes must be able to return False (#1173).

    ``rep_fusion`` used to be built unconditionally on every instance, so
    ``hasattr(gen, "rep_fusion")`` -- the test two ``diffusion.py`` call sites
    used -- was a constant ``True``. A probe that cannot fail is a facade, not a
    check (CLAUDE.md pitfall #16).

    These tests invoke the REAL property objects off the class via ``.fget``
    against a stub state, so they exercise production code without constructing
    the generator (which is heavy). The property function under test is the one
    that ships.
    """

    def test_supports_repetition_fusion_is_false_when_not_built(self) -> None:
        from types import SimpleNamespace

        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceColdDiffusionGenerator,
        )

        prop = KSpaceColdDiffusionGenerator.supports_repetition_fusion
        assert prop.fget(SimpleNamespace(rep_fusion=None)) is False

    def test_supports_repetition_fusion_is_true_when_built(self) -> None:
        """PLANT: the other polarity. A probe pinned only on its False leg would
        pass against a property hardwired to return False."""
        from types import SimpleNamespace

        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceColdDiffusionGenerator,
        )

        prop = KSpaceColdDiffusionGenerator.supports_repetition_fusion
        assert prop.fget(SimpleNamespace(rep_fusion=object())) is True

    def test_supports_5d_input_does_not_depend_on_rep_fusion(self) -> None:
        """The invariant the old predicate conflated (non-negotiable 17).

        5D consumption is provided by the ``FourierBridgeNetwork`` backbone and
        has nothing to do with NEX fusion, so the answer must not move when
        ``rep_fusion`` does.
        """
        from types import SimpleNamespace

        from spectramr.models.generators.kspace_cold_diffusion_generator import (
            KSpaceColdDiffusionGenerator,
        )

        prop = KSpaceColdDiffusionGenerator.supports_5d_input
        without = prop.fget(SimpleNamespace(rep_fusion=None))
        with_ = prop.fget(SimpleNamespace(rep_fusion=object()))
        assert without is with_ is True

    def test_rep_fusion_is_built_only_under_a_conditional(self) -> None:
        """Structural gate: the construction must stay inside an ``if``.

        This is the regression that #1173 is: an unconditional build makes both
        probes above unanswerable no matter how they are written. Checked on the
        AST rather than by constructing the model, which the local machine
        cannot afford.
        """
        import ast
        import inspect

        from spectramr.models.generators import kspace_cold_diffusion_generator as mod

        tree = ast.parse(inspect.getsource(mod))
        gen_cls = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "KSpaceColdDiffusionGenerator"
        )
        init = next(
            n
            for n in ast.walk(gen_cls)
            if isinstance(n, ast.FunctionDef) and n.name == "__init__"
        )

        # Every ComplexRepetitionFusion(...) call inside __init__ must sit under
        # an `if` that tests num_repetitions.
        guarded_calls = 0
        total_calls = 0
        for node in ast.walk(init):
            if isinstance(node, ast.If):
                cond = ast.unparse(node.test)
                if "num_repetitions" not in cond:
                    continue
                for inner in ast.walk(node):
                    if (
                        isinstance(inner, ast.Call)
                        and getattr(inner.func, "id", None) == "ComplexRepetitionFusion"
                    ):
                        guarded_calls += 1
        for node in ast.walk(init):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "ComplexRepetitionFusion"
            ):
                total_calls += 1

        assert total_calls >= 1, "construction site vanished; update this gate"
        assert guarded_calls == total_calls, (
            f"{total_calls - guarded_calls} ComplexRepetitionFusion construction(s) "
            "are not guarded by a num_repetitions conditional -- the capability "
            "probes become constants again (#1173)."
        )


class TestDeviceForwardedToUndersamplingProcess:
    """``device`` is declared, not swallowed by ``**kwargs`` (#1508).

    The parameter exists so ``resolve_generator_kwargs`` step 3d can inject the
    run's resolved device: that injection is gated on EXPLICIT acceptance
    precisely because almost every other generator declares ``**kwargs``, and
    injecting into all of them would leak an unexpected key into strict
    sub-configs. Declaring it here is what makes this the one class in the
    registry the injection reaches.
    """

    @staticmethod
    def _model(**kwargs) -> KSpaceColdDiffusionGenerator:
        return KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=16,
            num_layers=2,
            attention_type="none",
            num_timesteps=8,
            **kwargs,
        )

    def test_named_explicitly_in_the_signature(self) -> None:
        """Pinned on the signature, not on behaviour: dropping the parameter
        would let ``device`` fall into ``**kwargs`` -- still constructing fine,
        still never reaching the process -- and step 3d would stop injecting it
        because the contract is read off this signature."""
        import inspect

        params = inspect.signature(KSpaceColdDiffusionGenerator.__init__).parameters
        assert "device" in params

    def test_declared_device_reaches_the_process(self) -> None:
        model = self._model(device="cuda")
        assert model.kspace_process.mask_device == torch.device("cuda")
        assert model.kspace_process.mask_generator.device.type == "cuda"

    def test_default_leaves_the_process_on_cpu(self) -> None:
        model = self._model()
        assert model.kspace_process.mask_device is None
        assert model.kspace_process.mask_generator.device.type == "cpu"


# ---------------------------------------------------------------------------
# DC noise levels are wired; dc_weight is inert under hard (#1525)
# ---------------------------------------------------------------------------


def _dc_generator(**dc_kwargs):
    return KSpaceColdDiffusionGenerator(
        in_channels=2,
        out_channels=2,
        base_channels=16,
        num_layers=2,
        attention_type="none",
        **dc_kwargs,
    )


def test_hard_dc_layer_receives_the_declared_noise_levels():
    """Before #1525 these never arrived: the layer was built with a bare ``()``."""
    from spectramr.infrastructure.physics.data_consistency import HardDataConsistency

    model = _dc_generator(
        dc_method="hard", train_noise_level=0.077, eval_noise_level=0.033
    )
    assert isinstance(model.dc_layer, HardDataConsistency)
    assert model.dc_layer.train_noise_level == pytest.approx(0.077)
    assert model.dc_layer.eval_noise_level == pytest.approx(0.033)


def test_hard_dc_noise_levels_default_when_undeclared():
    model = _dc_generator(dc_method="hard")
    assert model.dc_layer.train_noise_level == pytest.approx(0.01)
    assert model.dc_layer.eval_noise_level == pytest.approx(0.005)


def test_hard_dc_ignores_dc_weight_by_construction():
    """``dc_weight`` is inert under ``hard`` -- and MUST stay that way.

    Honouring it would silently convert hard DC (which REPLACES the acquired
    bins) into soft DC (which blends toward them). The layer is asserted to be
    byte-identical across two very different declared weights.
    """
    import torch

    a = _dc_generator(dc_method="hard", dc_weight=0.5)
    b = _dc_generator(dc_method="hard", dc_weight=42.0)
    assert type(a.dc_layer) is type(b.dc_layer)
    assert not any(p.requires_grad for p in a.dc_layer.parameters())
    assert list(a.dc_layer.parameters()) == list(b.dc_layer.parameters()) == []

    recon = torch.randn(1, 2, 8, 8)
    obs = torch.randn(1, 2, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., ::2] = 1.0
    torch.manual_seed(0)
    out_a = a.dc_layer(recon, obs, mask, is_kspace_domain=True)
    torch.manual_seed(0)
    out_b = b.dc_layer(recon, obs, mask, is_kspace_domain=True)
    assert torch.equal(out_a, out_b), "dc_weight changed hard DC's output"


def test_unsupported_noise_type_raises_through_the_generator():
    """No silent fallback: an unreachable noise model must not degrade to Gaussian."""
    with pytest.raises(ValueError, match="unsupported noise_type"):
        _dc_generator(dc_method="hard", noise_type="rician")


# --------------------------------------------------------------------------
# `return_pre_dc`: exposing the pre-data-consistency proposal in eval mode.
#
# Under `dc_method: hard` the reverse process never evaluates the model at t=0,
# and after DC at t=0 every bin is acquired, so the output IS the measurement.
# The only measurable thing at the terminal rung is the proposal the network
# makes BEFORE data consistency overwrites it -- which is also the only place a
# gradient exists there (`lambda_pre_dc_kspace`). These pin that the flag hands
# back that proposal and not the post-DC output, because the post-DC output
# would score as a flawless reconstruction while measuring nothing.
#
# `eval_noise_level=0.0` throughout: hard DC otherwise adds N(0, 0.005) to the
# measurement in eval, which makes these comparisons non-deterministic (#1689).
# --------------------------------------------------------------------------


def _pre_dc_generator():
    return _dc_generator(dc_method="hard", eval_noise_level=0.0).eval()


def _pre_dc_forward_kwargs(x: torch.Tensor) -> dict:
    kwargs = _measurement_kwargs(x)
    kwargs["mask"] = torch.ones_like(kwargs["mask"])  # t=0: every bin acquired
    kwargs["kspace_measured"] = torch.randn_like(x)
    kwargs["sensitivity_maps"] = torch.randn_like(x)
    return kwargs


def test_return_pre_dc_is_keyword_only_so_a_typo_cannot_vanish_into_kwargs():
    import inspect

    param = inspect.signature(KSpaceColdDiffusionGenerator.forward).parameters[
        "return_pre_dc"
    ]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is False


def test_generator_declares_that_it_exposes_pre_dc():
    # The probe reads this off the CLASS to decide whether to emit its metric
    # keys; `train.py:_all_reduce_val_metrics` packs sorted keys positionally,
    # so that decision has to be rank-invariant.
    assert KSpaceColdDiffusionGenerator.exposes_pre_dc is True


def test_eval_forward_without_the_flag_still_returns_a_bare_tensor():
    model = _pre_dc_generator()
    x = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        out = model(x, torch.zeros(1, dtype=torch.long), **_pre_dc_forward_kwargs(x))
    assert isinstance(out, torch.Tensor)


def test_eval_forward_with_the_flag_returns_the_pair():
    model = _pre_dc_generator()
    x = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        out = model(
            x,
            torch.zeros(1, dtype=torch.long),
            return_pre_dc=True,
            **_pre_dc_forward_kwargs(x),
        )
    assert isinstance(out, tuple) and len(out) == 2
    assert all(isinstance(t, torch.Tensor) for t in out)


def test_pre_dc_element_is_not_the_post_dc_output():
    """The pin that matters: element [1] must be the proposal, not the result.

    With every bin acquired, hard DC replaces the whole prediction, so the two
    are numerically far apart. Returning `x_out` twice -- the plausible wiring
    mistake -- makes this assertion fail.
    """
    model = _pre_dc_generator()
    x = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        post_dc, pre_dc = model(
            x,
            torch.zeros(1, dtype=torch.long),
            return_pre_dc=True,
            **_pre_dc_forward_kwargs(x),
        )
    assert pre_dc is not post_dc
    assert not torch.allclose(pre_dc, post_dc), (
        "pre-DC and post-DC are identical; the probe would report the "
        "measurement back to itself as a perfect reconstruction"
    )


def test_post_dc_at_t0_reproduces_the_measurement_but_pre_dc_does_not():
    """Names what each element means, rather than only that they differ.

    At t=0 with a full mask, hard DC writes the measurement into every bin, so
    the post-DC output IS the measurement. The pre-DC proposal is the network's
    own answer and is not. This is what makes the terminal rung unmeasurable
    post-DC and measurable pre-DC.
    """
    model = _pre_dc_generator()
    x = torch.randn(1, 2, 16, 16)
    kwargs = _pre_dc_forward_kwargs(x)
    measured = kwargs["kspace_measured"]
    with torch.no_grad():
        post_dc, pre_dc = model(
            x, torch.zeros(1, dtype=torch.long), return_pre_dc=True, **kwargs
        )
    post_err = (post_dc - measured).abs().max()
    pre_err = (pre_dc - measured).abs().max()
    assert post_err < pre_err, (
        f"post-DC should track the measurement more closely than the raw "
        f"proposal does (post={post_err:.4g}, pre={pre_err:.4g})"
    )


def test_training_mode_still_returns_the_pair_without_the_flag():
    """The pre-existing training contract is unchanged by the new keyword."""
    model = _dc_generator(dc_method="hard", eval_noise_level=0.0).train()
    x = torch.randn(1, 2, 16, 16)
    out = model(x, torch.zeros(1, dtype=torch.long), **_pre_dc_forward_kwargs(x))
    assert isinstance(out, tuple) and len(out) == 2


class TestSamplerDeterminismKnobs:
    """``sampler_sigma`` / ``sampler_seed`` / ``selection_rule`` model_kwargs (issue #1286).

    Read, validated at BUILD, stamped on the generator, and -- the half that was
    missing -- forwarded to the ``cold_mri`` sampler so the reverse loop runs at
    the sigma the YAML declares. ``seed_offset`` (the validation ensemble's
    per-member stream) follows ``start_timestep``'s forwarding rule, except that a
    non-zero offset the sampler cannot take RAISES.
    """

    @staticmethod
    def _build(**kw) -> KSpaceColdDiffusionGenerator:
        return KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            attention_type="none",
            kspace_log_scaled=False,
            **kw,
        )

    @staticmethod
    def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
        measurement = torch.randn(1, 2, 16, 16)
        mask = torch.zeros(1, 1, 16, 16)
        mask[..., ::2] = 1.0
        return measurement * mask, mask

    def test_defaults_are_the_deterministic_sampler(self) -> None:
        model = self._build()
        assert model.sampler_sigma == 0.0
        assert model.sampler_seed is None
        assert model._selection_rule == "fixed"

    def test_knobs_are_read_and_stamped(self) -> None:
        model = self._build(sampler_sigma=0.2, sampler_seed=5)
        assert model.sampler_sigma == pytest.approx(0.2)
        assert model.sampler_seed == 5

    def test_negative_sigma_raises_at_build(self) -> None:
        with pytest.raises(ValueError, match="sampler_sigma"):
            self._build(sampler_sigma=-0.1)

    def test_unknown_selection_rule_raises_at_build(self) -> None:
        with pytest.raises(ValueError, match="selection_rule"):
            self._build(selection_rule="greedy")

    def test_knobs_reach_get_sampler_on_the_cold_mri_path(self, monkeypatch) -> None:
        """The planted #1286 shape: the values must be IN the kwargs the registry gets."""
        import spectramr.models.diffusion.samplers as samplers

        seen: dict = {}

        class _Sampler:
            def sample(self, measurement, mask, **kwargs):
                return measurement

        def _fake_get_sampler(name, **kwargs):
            seen["name"] = name
            seen.update(kwargs)
            return _Sampler()

        monkeypatch.setattr(samplers, "get_sampler", _fake_get_sampler)
        measurement, mask = self._inputs()

        self._build(sampler_sigma=0.3, sampler_seed=9).sample(measurement, mask=mask)

        assert seen["name"] == "cold_mri"
        assert seen["sampler_sigma"] == pytest.approx(0.3)
        assert seen["sampler_seed"] == 9
        assert seen["selection_rule"] == "fixed"

    def test_seed_offset_is_forwarded_when_the_sampler_declares_it(self, monkeypatch) -> None:
        import spectramr.models.diffusion.samplers as samplers

        received: list = []

        class Accepting:
            def sample(self, measurement, mask, seed_offset=0):
                received.append(seed_offset)
                return measurement

        monkeypatch.setattr(samplers, "get_sampler", lambda name, **kw: Accepting())
        measurement, mask = self._inputs()

        self._build().sample(measurement, mask=mask, seed_offset=2)

        assert received == [2]

    def test_a_zero_offset_is_never_forwarded(self, monkeypatch) -> None:
        """Member 0 and every single-sample call stay the legacy call, byte for byte."""
        import spectramr.models.diffusion.samplers as samplers

        seen: list = []

        class Strict:
            def sample(self, measurement, mask, **kwargs):
                seen.append(dict(kwargs))
                return measurement

        monkeypatch.setattr(samplers, "get_sampler", lambda name, **kw: Strict())
        measurement, mask = self._inputs()
        model = self._build()

        model.sample(measurement, mask=mask)
        model.sample(measurement, mask=mask, seed_offset=0)

        assert seen == [{}, {}]

    def test_a_nonzero_offset_the_sampler_cannot_take_raises(self, monkeypatch) -> None:
        """Warning here would make member 1 replay member 0's stream in silence."""
        import spectramr.models.diffusion.samplers as samplers

        class Legacy:
            def sample(self, measurement, mask, start_timestep=None):
                return measurement

        monkeypatch.setattr(samplers, "get_sampler", lambda name, **kw: Legacy())
        measurement, mask = self._inputs()

        with pytest.raises(ValueError, match="seed_offset"):
            self._build().sample(measurement, mask=mask, seed_offset=1)


# ─────────────────────────────────────────────────────────────────────────────
# Every attention_type the dispatch builds must be able to LEAVE its identity.
#
# The #471 wrapper made all eight blocks start at rho = 1.000; three of them
# (self / kernelized / sparse) already zero-initialised their own output
# projection, and the product of the two mechanisms is a fixed point with zero
# gradient everywhere. This test drives the production dispatch rather than a
# hand-composed wrapper, because that composition is what the arms build.
# ─────────────────────────────────────────────────────────────────────────────

# One case per attention_type the dispatch can build, not a sample: the saddle was
# a property of the COMPOSITION, so a type is only covered once it has been watched
# training off its own initialisation (non-negotiable 15).
_DISPATCH_ATTENTION_TYPES = [
    "self",
    "kernelized",
    "sparse",
    "channel",
    "dual_domain",
    "wavelet_freq",
    "kan_dual_domain",
    "null_space_dual_domain",
]


@pytest.mark.parametrize("attention_type", _DISPATCH_ATTENTION_TYPES)
def test_dispatch_attention_trains_off_its_identity_init(attention_type: str) -> None:
    """gamma must move on backward 1 and the inner block must train on backward 2."""
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceDownsampleBlock,
    )

    torch.manual_seed(0)
    block = KSpaceDownsampleBlock(
        64,
        64,
        attention_type=attention_type,
        use_complex_conv=True,
        time_embedding_dim=256,
        feature_domain="kspace",
    )
    attention = block.attention
    opt = torch.optim.AdamW(block.parameters(), lr=1e-2)
    target = torch.randn(1, 64, 16, 16)

    acquisition = torch.zeros(1, 1, 32, 32)
    acquisition[..., ::4] = 1.0
    acquisition[..., 14:18] = 1.0

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        out, _ = block(torch.randn(1, 64, 32, 32), torch.randn(1, 256), acquisition)
        ((out - target) ** 2).mean().backward()
        opt.step()

    step()
    assert attention.gamma.grad.abs().item() > 0.0, (
        f"attention_type={attention_type!r} gives gamma no gradient: the arm is "
        "bit-identical to attention_none for the whole run"
    )
    step()
    inner_grad = max(
        0.0 if p.grad is None else p.grad.abs().max().item()
        for p in attention.inner.parameters()
    )
    assert inner_grad > 0.0, (
        f"attention_type={attention_type!r} never delivers gradient to its own "
        "parameters, so the mechanism the arm varies is never learned"
    )


def test_dispatch_opts_the_three_blocks_out_of_their_own_zero_init() -> None:
    """The election itself (non-negotiable 17), not just its effect.

    Asserting only "gamma trains" would stay green if someone restored the
    second mechanism and compensated elsewhere; this pins which owner won.
    """
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceDownsampleBlock,
        KSpaceUpsampleBlock,
    )

    for attention_type, proj_attr in [
        ("self", "proj"),
        ("kernelized", "out_proj"),
        ("sparse", "proj"),
    ]:
        for builder, args in [(KSpaceDownsampleBlock, (64, 64)), (KSpaceUpsampleBlock, (64, 64))]:
            block = builder(
                *args,
                attention_type=attention_type,
                use_complex_conv=True,
                time_embedding_dim=256,
                feature_domain="kspace",
            )
            proj = getattr(block.attention.inner, proj_attr)
            assert proj.weight.abs().sum().item() > 0.0, (
                f"{builder.__name__}/{attention_type}: the block still zero-inits its "
                "own output projection under IdentityAtInitAttention"
            )


# ─────────────────────────────────────────────────────────────────────────────
# null_space_dual_domain is REACHED and FED, not merely registered (#16)
#
# The block raises without a mask, and the reverse sampler re-enters
# ``forward(x_t, t)`` with no kwargs at all -- the same hole the S-map stash was
# opened for. Registering the type and threading the training forward would
# still leave every validation event dead, so both paths are observed here.
# ─────────────────────────────────────────────────────────────────────────────


def _null_space_generator(size: int = 32):
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        KSpaceColdDiffusionGenerator,
    )

    torch.manual_seed(0)
    return KSpaceColdDiffusionGenerator(
        in_channels=8,
        out_channels=8,
        base_channels=16,
        num_res_blocks=4,
        attention_type="null_space_dual_domain",
        backbone_type="complex_unet",
        force_pure_kspace=True,
        use_complex_conv=True,
        activation="complex",
        time_embedding_dim=64,
        timesteps=8,
        sampling_steps=8,
        dc_method="hard",
        reverse_sampling_mode="replace_freeze_dc",
        kspace_log_scaled=False,
        base_acceleration=1.0,
        max_acceleration=8.0,
    ).eval()


def _spy_on_null_space_blocks(generator) -> list:
    """Record the (feature grid, mask) every null-space block is called with."""
    from spectramr.models.blocks.null_space_attention import NullSpaceDualDomainAttention

    calls: list = []
    for module in generator.modules():
        if isinstance(module, NullSpaceDualDomainAttention):
            module.register_forward_pre_hook(
                lambda mod, args, kwargs, sink=calls: sink.append(
                    (tuple(args[0].shape[-2:]), kwargs.get("mask"))
                ),
                with_kwargs=True,
            )
    return calls


def _acquisition(size: int):
    mask = torch.zeros(1, 1, size, size)
    mask[..., ::4] = 1.0
    mask[..., size // 2 - 2 : size // 2 + 2] = 1.0
    return mask, torch.randn(1, 4, size, size, dtype=torch.complex64)


def test_null_space_blocks_are_built_and_fed_on_the_training_forward() -> None:
    size = 32
    generator = _null_space_generator(size)
    calls = _spy_on_null_space_blocks(generator)
    mask, smaps = _acquisition(size)
    x = torch.randn(1, 8, size, size)

    with torch.no_grad():
        generator(
            x,
            timesteps=torch.full((1,), 3, dtype=torch.long),
            mask=mask,
            kspace_measured=x * mask,
            smaps=smaps,
        )

    assert calls, "no null-space block ran: the dispatch built something else"
    assert all(m is not None for _, m in calls), "a block ran without its mask"
    # The encoder downsamples by center-cropping k-space, so the blocks must see
    # more than one grid while the mask stays full-resolution for all of them.
    assert len({grid for grid, _ in calls}) > 1
    assert {tuple(m.shape[-2:]) for _, m in calls} == {(size, size)}


def test_the_reverse_sampler_still_feeds_the_mask_and_restores_the_stash() -> None:
    """The bare ``(x, t)`` re-entry is where a forward-signature-only route dies."""
    size = 32
    generator = _null_space_generator(size)
    calls = _spy_on_null_space_blocks(generator)
    mask, smaps = _acquisition(size)
    generator.set_current_smaps(smaps)

    with torch.no_grad():
        generator.sample(torch.randn(1, 8, size, size) * mask, mask=mask, inference_timesteps=3)

    assert calls, "no null-space block ran during reverse sampling"
    assert all(m is not None for _, m in calls), "a reverse step ran without the mask"
    assert generator._current_mask is None, "sample() must restore the previous stash"


#: The smallest arm that builds the ``complex_unet`` decoder these two classes
#: exercise. Module level rather than a class attribute so the two share one.
_BAND_BASE = {
    "in_channels": 4,
    "out_channels": 4,
    "base_channels": 8,
    "num_layers": 2,
    "use_complex_conv": True,
    "attention_type": "none",
    "force_pure_kspace": True,
    "backbone_type": "complex_unet",
    "condition_with_smaps": False,
    "use_dc": False,
    "kspace_log_scaled": False,
    "num_timesteps": 29,
}


class TestRadialBandTokensPlumbing:
    """The token bank reaches the decoder, and refuses a backbone without the seam.

    The reverse sampler calls ``model(x, t)`` bare, which is what leaves the
    generator's own DC layer inert at validation. This mechanism is rung-indexed
    rather than mask-indexed precisely so that call still carries what it needs,
    and that is asserted here rather than argued.
    """

    def _generator(self, **extra):
        torch.manual_seed(0)
        return KSpaceColdDiffusionGenerator(**_BAND_BASE, **extra).eval()

    def _tokens(self, model):
        return model.backbone.backbone.radial_band_tokens

    def test_the_knob_is_off_by_default(self):
        assert self._tokens(self._generator()) is None

    def test_the_rung_count_is_derived_from_the_process_not_declared(self):
        """One owner: ``model_kwargs.timesteps`` already states how many rungs exist."""
        assert self._tokens(self._generator(radial_band_tokens_bands=8)).n_rungs == 29

    def test_declaring_the_rung_count_raises(self):
        with pytest.raises(ValueError, match="derived from the process"):
            self._generator(radial_band_tokens_bands=8, radial_band_tokens_rungs=29)

    def test_a_misspelled_knob_raises_at_the_generator(self):
        """Planted: ``**kwargs`` plus no OPTION_SCHEMA means nothing else sees it."""
        with pytest.raises(ValueError, match="Unrecognised radial band token"):
            self._generator(radial_band_tokens_bnads=8)

    @pytest.mark.parametrize(
        "extra",
        [
            {"backbone_type": "unet", "attention_type": "none"},
            {"force_pure_kspace": False},
        ],
    )
    def test_a_backbone_without_the_seam_raises(self, extra):
        with pytest.raises(ValueError, match="radial_band_tokens_bands"):
            torch.manual_seed(0)
            KSpaceColdDiffusionGenerator(**{**_BAND_BASE, **extra}, radial_band_tokens_bands=8)

    def test_it_fires_on_the_bare_reverse_loop_call(self):
        """``model(x, t)`` with no kwargs is where the reported numbers come from."""
        model = self._generator(radial_band_tokens_bands=8)
        fired: list[int] = []
        self._tokens(model).register_forward_hook(lambda m, i, o: fired.append(int(i[1][0])))
        with torch.no_grad():
            model(torch.randn(1, 4, 32, 32), torch.full((1,), 22, dtype=torch.long))
        assert len(fired) == len(model.backbone.backbone.ups)
        assert set(fired) == {22}, "the rung must be the timestep the sampler passed"

    def test_an_armed_arm_reproduces_its_control_at_initialisation(self):
        x, t = torch.randn(1, 4, 32, 32), torch.tensor([7])
        with torch.no_grad():
            control = self._generator()(x, t)
            armed = self._generator(radial_band_tokens_bands=8)(x, t)
        assert torch.equal(control, armed)


class TestRadialBandGainTimestepConditioning:
    """The gain's timestep half was declared by the class and never passed.

    ``RadialBandGain`` accepts ``time_embed_dim`` and ``band_gains`` consumes a
    ``time_embedding``, but the generator built it with neither, so the advertised
    ``[annulus x timestep]`` conditioning ran as ``[annulus]`` whatever an arm
    asked for.
    """

    def _generator(self, **extra):
        torch.manual_seed(0)
        return KSpaceColdDiffusionGenerator(**_BAND_BASE, **extra).eval()

    def test_the_width_is_zero_unless_an_arm_declares_it(self):
        """No existing arm moves: absent still means the per-annulus scalar."""
        assert self._generator(radial_band_gain_bands=8).radial_band_gain.time_embed_dim == 0

    def test_declaring_the_width_builds_a_time_conditioned_gain(self):
        model = self._generator(radial_band_gain_bands=8, radial_band_gain_time_embed_dim=32)
        assert model.radial_band_gain.time_embed_dim == 32

    def test_the_embedding_actually_reaches_band_gains(self):
        """Observed at the seam: the class already raises on a missing embedding,
        so the thing worth asserting is that one arrives at all."""
        model = self._generator(radial_band_gain_bands=8, radial_band_gain_time_embed_dim=32)
        seen: list[bool] = []
        inner = model.radial_band_gain.band_gains

        def spy(kspace, time_embedding=None):
            seen.append(time_embedding is not None and time_embedding.shape[-1] == 32)
            return inner(kspace, time_embedding)

        model.radial_band_gain.band_gains = spy
        with torch.no_grad():
            model(torch.randn(1, 4, 32, 32), torch.tensor([3]))
        assert seen == [True]

    def test_a_time_conditioned_gain_still_reproduces_its_control_at_init(self):
        """``gamma`` is still zero, so the A/B stays attributable."""
        x, t = torch.randn(1, 4, 32, 32), torch.tensor([3])
        with torch.no_grad():
            control = self._generator()(x, t)
            gained = self._generator(radial_band_gain_bands=8, radial_band_gain_time_embed_dim=32)(
                x, t
            )
        assert torch.allclose(control, gained, atol=1e-6)


# ── the phase-safe attention query must carry both halves ────────────────────
class TestPhaseSafeGuidanceLayout:
    """The bridge's guidance is interleaved, like everything else it feeds.

    It was built as a BLOCKED ``[Re...,Im...]`` stack while
    ``PhaseSafeDualAttention`` reads ``0::2``/``1::2``. With S-map conditioning
    the stack is twice the value width, so the truncation to the value's channel
    count kept exactly the real parts and dropped every imaginary one — the
    query was ``Re(ifft2c(x))`` and nothing else, with no shape error to show
    for it. All nine ``attention_type: dual_domain`` arms take this path.
    """

    @staticmethod
    def _query_on_the_control_arm():
        import contextlib

        import torch as _t

        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

        arm = "experiments/inprogress/kspace_filling/experiment_11_kspace_cold_diffusion.yaml"
        cfg = TrainingSettings.from_yaml(arm)
        _t.manual_seed(0)
        gen = ModelBuilder(config=cfg, device="cpu").build_generator().validate().build()
        gen = gen["generator"].eval()
        attention = getattr(gen.backbone, "phase_safe_attention", None)
        assert attention is not None, "the dual_domain arm lost its bridge attention"
        captured: dict[str, object] = {}
        attention.register_forward_pre_hook(
            lambda m, i: captured.update(q=i[1].detach().clone())
        )
        channels = cfg.model.in_channels
        # The post-backbone DC guard fires later; the query is already captured.
        with _t.no_grad(), contextlib.suppress(Exception):
            gen(
                _t.randn(1, channels, 64, 64),
                _t.zeros(1, dtype=_t.long),
                sensitivity_maps=_t.randn(1, max(channels // 2, 1), 64, 64, dtype=_t.complex64),
            )
        assert "q" in captured, "the attention never fired"
        return captured["q"]

    def test_the_query_carries_the_imaginary_half(self):
        query = self._query_on_the_control_arm()
        assert float(query[:, 1::2].abs().sum()) > 1.0, (
            "the odd (imaginary) channels are empty; the guidance is blocked again"
        )

    def test_the_query_still_carries_the_real_half(self):
        """Guards the check above from passing on a query that lost the real part."""
        query = self._query_on_the_control_arm()
        assert float(query[:, 0::2].abs().sum()) > 1.0

    def test_the_guidance_is_built_with_the_layout_ssot(self):
        import inspect

        from spectramr.models.generators import kspace_cold_diffusion_generator as mod

        src = inspect.getsource(mod.FourierBridgeNetwork.forward)
        assert "complex_to_interleaved(spatial_complex)" in src
        assert "torch.cat([spatial_complex.real, spatial_complex.imag], dim=1)" not in src
