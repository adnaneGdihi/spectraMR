import pytest
import torch

from spectramr.models.generators.complex_unet import ComplexUNet


def test_complex_unet_shapes():
    """Verify that ComplexUNet can handle forward pass with different depths."""
    batch_size = 2
    in_channels = 4  # 2 complex
    out_channels = 4  # 2 complex
    img_size = (64, 64)

    # Test with standard features
    features = (16, 32, 64)
    model = ComplexUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        img_size=img_size,
    )

    x = torch.randn(batch_size, in_channels, *img_size)
    timesteps = torch.tensor([0, 10])

    output = model(x, timesteps=timesteps)

    assert output.shape == (batch_size, out_channels, *img_size)


def test_complex_unet_depth_variations():
    """Verify ComplexUNet with different feature depths."""
    batch_size = 1
    in_channels = 4
    out_channels = 4
    img_size = (32, 32)

    for depth in [1, 2, 4]:
        features = tuple(16 * (2**i) for i in range(depth))
        model = ComplexUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            img_size=img_size,
        )
        x = torch.randn(batch_size, in_channels, *img_size)
        output = model(x)
        assert output.shape == (batch_size, out_channels, *img_size)


class TestFeatureDomainThreading:
    """The feature_domain kwarg must reach every domain-aware sub-block."""

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

    def test_default_feature_domain_is_kspace(self):
        model = ComplexUNet(
            in_channels=4,
            out_channels=4,
            features=(16, 32),
            attention_type="kan_dual_domain",
        )
        assert self._domains(model) == {"kspace"}

    def test_image_feature_domain_threaded_to_all_blocks(self):
        model = ComplexUNet(
            in_channels=4,
            out_channels=4,
            features=(16, 32),
            attention_type="kan_dual_domain",
            feature_domain="image",
        )
        # both the unconditional DualDomainBlock and the KAN attention
        assert self._domains(model) == {"image"}

    def test_invalid_feature_domain_raises(self):
        with pytest.raises(ValueError, match="feature_domain"):
            ComplexUNet(
                in_channels=4,
                out_channels=4,
                features=(16, 32),
                feature_domain="frequency",
            )


class TestKSpaceFeatureNorm:
    """Opt-in inter-layer normalization (the experiment_11 divergence fix)."""

    def _norms(self, model):
        return [
            model.norm_initial,
            *model.norm_downs,
            model.norm_bottleneck,
            *model.norm_ups,
        ]

    def test_default_is_identity_and_non_disturbing(self):
        """Absent/``none`` builds Identity norms => backbone byte-identical."""
        import torch.nn as nn

        model = ComplexUNet(in_channels=4, out_channels=4, features=(16, 32))
        assert model.kspace_feature_norm == "none"
        assert all(isinstance(n, nn.Identity) for n in self._norms(model))

    def test_rms_builds_complexrmsnorm_at_every_stage(self):
        from spectramr.models.layers.complex_norm import ComplexRMSNorm

        model = ComplexUNet(
            in_channels=4,
            out_channels=4,
            features=(16, 32, 64),
            kspace_feature_norm="rms",
        )
        norms = self._norms(model)
        assert all(isinstance(n, ComplexRMSNorm) for n in norms)
        # channel counts follow the stage geometry: initial, downs, bottleneck, ups
        assert model.norm_initial.complex_channels == 16
        assert model.norm_bottleneck.complex_channels == 64 * 2

    def test_rms_forward_shape_matches_none(self):
        x = torch.randn(2, 4, 32, 32)
        ts = torch.tensor([0, 10])
        base = ComplexUNet(in_channels=4, out_channels=4, features=(16, 32))
        normed = ComplexUNet(
            in_channels=4,
            out_channels=4,
            features=(16, 32),
            kspace_feature_norm="rms",
        )
        assert normed(x, timesteps=ts).shape == base(x, timesteps=ts).shape

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="kspace_feature_norm"):
            ComplexUNet(
                in_channels=4,
                out_channels=4,
                features=(16, 32),
                kspace_feature_norm="groupnorm",
            )


if __name__ == "__main__":
    pytest.main([__file__])


# ─────────────────────────────────────────────────────────────────────────────
# Identity-at-init across the WHOLE advertised set (issue #471)
#
# This is the test that was missing. Per-block tests existed for the residual
# family, so `self`/`kernelized`/`sparse` were pinned at rho = 1.0 while
# `channel`, `dual_domain`, `wavelet_freq` and `kan_dual_domain` drifted to
# 0.65 / 1.05 / 0.54 / 8.5 unnoticed. Parametrising over the registry set means a
# newly added attention type is covered the moment it is advertised.
# ─────────────────────────────────────────────────────────────────────────────

_BLOCK_ATTENTION_TYPES = sorted(
    __import__(
        "spectramr.models.blocks.attention_domains", fromlist=["x"]
    ).COMPLEX_UNET_BLOCK_ATTENTION
)


def _acquisition_mask(H: int = 32, W: int = 32) -> "torch.Tensor":
    """Periodic phase-encode lines plus a central ACS band -- what the arms acquire."""
    import torch

    mask = torch.zeros(1, 1, H, W)
    mask[..., ::4] = 1.0
    mask[..., H // 2 - 2 : H // 2 + 2] = 1.0
    return mask


def _kspace_input(H: int = 32, W: int = 32) -> "torch.Tensor":
    import torch

    torch.manual_seed(0)
    ky = torch.fft.fftshift(torch.fft.fftfreq(H))[:, None]
    kx = torch.fft.fftshift(torch.fft.fftfreq(W))[None, :]
    amp = 1.0 / (ky**2 + kx**2).sqrt().clamp_min(1e-3)
    return (amp * torch.randn(1, 8, H, W)).contiguous()


@pytest.mark.parametrize("attention_type", _BLOCK_ATTENTION_TYPES)
def test_every_attention_type_is_identity_at_init(attention_type: str) -> None:
    """Every ``.attention`` site must report rho == 1.000 exactly on a fresh model.

    The call site is REPLACE (``x = self.attention(x)``), so a block that is not
    identity at init discards its input and emits its own output -- making
    ``attention_type`` vary the effective initialisation as well as the mechanism,
    which is not the hypothesis the shootout tests.
    """
    import torch

    from spectramr.models.generators.complex_unet import ComplexUNet

    torch.manual_seed(7)
    model = ComplexUNet(
        in_channels=8,
        out_channels=8,
        features=(16, 32),
        time_embedding_dim=256,
        img_size=(32, 32),
        padding_mode="circular",
        feature_domain="kspace",
        attention_type=attention_type,
        kspace_feature_norm="rms",
        kan_dual_domain_kwargs={"max_dense_attn_tokens": 256},
        num_contrasts=4,
        phase_safe_dim=64,
    ).eval()

    rhos: list[float] = []
    handles = []
    for name, mod in model.named_modules():
        if name.split(".")[-1] == "attention" and not isinstance(mod, torch.nn.Identity):
            handles.append(
                mod.register_forward_hook(
                    lambda _m, inp, out, sink=rhos: sink.append((out.norm() / inp[0].norm()).item())
                )
            )
    try:
        with torch.no_grad():
            # A mask is passed for EVERY type, not only the one that needs it: the
            # domain-agnostic blocks ignore the kwarg, and branching on the type here
            # would make the test agree with the dispatch by construction.
            model(_kspace_input(), torch.tensor([3]), mask=_acquisition_mask())
    finally:
        for h in handles:
            h.remove()

    if attention_type in ("none", "cross_contrast_olmpa"):
        # `none` has no block attention; cross_contrast_olmpa lives at the
        # bottleneck (`cc_olmpa`), covered in test_attention.py.
        assert rhos == []
        return

    assert rhos, f"{attention_type} built no attention site"
    for rho in rhos:
        assert rho == pytest.approx(1.0, abs=1e-6), (
            f"{attention_type}: rho={rho:.4f} at init; every block must start as an "
            f"exact identity so the shootout delta is the learned mechanism"
        )


def test_t_emb_reaches_time_conditioned_attention() -> None:
    """Wrapping must not blind the KAN block's timestep conditioning.

    The pre-fix forward dispatched t_emb on an isinstance check against a
    hardcoded class tuple; the wrapper makes that check false. Detection is now by
    signature, so assert the wrapper actually reports it.
    """
    import torch

    from spectramr.models.generators.complex_unet import ComplexUNet

    torch.manual_seed(7)
    model = ComplexUNet(
        in_channels=8,
        out_channels=8,
        features=(16, 32),
        time_embedding_dim=256,
        img_size=(32, 32),
        padding_mode="circular",
        feature_domain="kspace",
        attention_type="kan_dual_domain",
        kan_dual_domain_kwargs={"max_dense_attn_tokens": 256},
    )

    sites = [
        m
        for n, m in model.named_modules()
        if n.split(".")[-1] == "attention" and hasattr(m, "takes_t_emb")
    ]
    assert sites, "no wrapped attention sites found"
    assert all(m.takes_t_emb for m in sites)


def test_cc_olmpa_bottleneck_is_one_to_one_regardless_of_num_contrasts():
    """The bottleneck OLMPA has no contrast axis, and must not pretend to.

    The construction previously read ``num_contrasts = kwargs.pop("num_contrasts", 4)``
    and then passed the literal ``1``, so a declared value was consumed and
    discarded in the same breath. Every kspace_filling arm declares
    ``num_contrasts: 3``; the value is a data-level quantity and the block
    attends at bottleneck resolution where the source|target split is 1:1 by
    construction, so 1 is correct -- the defect was the misleading read, which
    grep reports as a wired knob.
    """
    common = dict(
        in_channels=4,
        out_channels=4,
        features=(8, 16),
        img_size=(32, 32),
        attention_type="cross_contrast_olmpa",
    )
    for declared in (1, 3, 4):
        model = ComplexUNet(num_contrasts=declared, **common)
        assert model.cc_olmpa is not None
        assert model.cc_olmpa.num_contrasts == 1, (
            f"num_contrasts={declared} leaked into the bottleneck block"
        )


def test_cc_olmpa_phase_safe_dim_is_honoured():
    """``phase_safe_dim`` IS a real parameter of this block, unlike num_contrasts."""
    model = ComplexUNet(
        in_channels=4,
        out_channels=4,
        features=(8, 16),
        img_size=(32, 32),
        attention_type="cross_contrast_olmpa",
        phase_safe_dim=64,
    )
    assert model.cc_olmpa.phase_safe_dim == 64


class TestPhaseEquivariance:
    """The backbone must commute with a global phase rotation of its input.

    A global phase offset on k-space is physically meaningless -- it is a choice
    of receiver reference, not information -- so ``f(e^{i.phi} x) = e^{i.phi} f(x)``
    should hold exactly. These pin the property that measurement shows is already
    true, so a future edit cannot lose it silently.

    Measured across all eight shootout attention types at 32x32, features
    (32, 64): deviation 0.0000 without time conditioning, 0.0042-0.0062 WITH it.
    The whole gap is ComplexTimeInjection, which adds a real-projected embedding
    (``x + t_emb``) whose value does not rotate with the input. Conv bias is NOT
    the mechanism -- the built model has 53 ComplexConv2d modules and zero of
    them carry a bias parameter. The conditioning question is a design change,
    not a bug, and is tracked separately.
    """

    @staticmethod
    def _rotate(z: torch.Tensor, phi: float) -> torch.Tensor:
        """Rotate a real-interleaved complex tensor by ``phi``."""
        cos, sin = torch.cos(torch.tensor(phi)), torch.sin(torch.tensor(phi))
        real, imag = z[:, 0::2], z[:, 1::2]
        out = torch.empty_like(z)
        out[:, 0::2] = cos * real - sin * imag
        out[:, 1::2] = sin * real + cos * imag
        return out

    @staticmethod
    def _model(attention_type: str) -> ComplexUNet:
        torch.manual_seed(0)
        return ComplexUNet(
            in_channels=8,
            out_channels=8,
            features=(32, 64),
            img_size=(32, 32),
            attention_type=attention_type,
            activation="complex",
        ).eval()

    @pytest.mark.parametrize(
        "attention_type",
        [
            "none",
            "channel",
            "self",
            "sparse",
            "kernelized",
            "wavelet_freq",
            "dual_domain",
            "kan_dual_domain",
        ],
    )
    def test_backbone_is_phase_equivariant_without_time_conditioning(self, attention_type):
        model = self._model(attention_type)
        x = torch.randn(1, 8, 32, 32)
        with torch.no_grad():
            baseline = model(x)
            rotated = model(self._rotate(x, 0.7))
        deviation = float(
            (rotated - self._rotate(baseline, 0.7)).abs().max()
            / baseline.abs().max().clamp(min=1e-12)
        )
        assert deviation < 1e-4, (
            f"{attention_type} broke phase equivariance (deviation {deviation:.2e}); "
            "a real-valued op or a bias term entered the complex path"
        )

    def test_no_complex_conv_carries_a_bias(self):
        """bias in k-space is a spatial Dirac delta (Hammernik et al. 2018).

        It is also the textbook phase-equivariance breaker, so this is the
        tripwire for both. Every ComplexConv2d in the built backbone must be
        bias-free -- asserted over the constructed module tree rather than by
        reading constructor arguments, because the defaults are set in several
        different call sites.
        """
        from spectramr.models.layers.complex_conv import ComplexConv2d

        model = self._model("none")
        convs = [m for m in model.modules() if isinstance(m, ComplexConv2d)]
        assert convs, "fixture built no ComplexConv2d — the assertion would be vacuous"
        biased = [
            name
            for name, module in model.named_modules()
            if isinstance(module, ComplexConv2d) and getattr(module, "bias", None) is not None
        ]
        assert not biased, f"ComplexConv2d modules carry a bias: {biased}"


class TestGradCheckpointing:
    """``set_grad_checkpointing`` must cut retained activations, not change math.

    The regression this guards: ``ModelBuilder`` only takes the native branch if
    the model exposes ``set_grad_checkpointing``. Without it the generic fallback
    matches ``nn.Conv2d``/``nn.Linear``/``nn.BatchNorm2d`` only, which on a
    ComplexConv2d network reaches ~3% of the parameter mass while still logging
    that checkpointing was applied.
    """

    @staticmethod
    def _model():
        return ComplexUNet(
            in_channels=4,
            out_channels=4,
            features=(16, 32, 64),
            img_size=(64, 64),
        )

    @staticmethod
    def _saved_bytes(model, x, timesteps):
        """Bytes of unique storage autograd stashes for backward."""
        seen: dict[int, int] = {}

        def pack(t):
            if isinstance(t, torch.Tensor):
                seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            model(x, timesteps=timesteps)
        return sum(seen.values())

    def test_default_is_off(self):
        """The un-opted-in forward must stay allocation-identical."""
        assert self._model().grad_checkpointing is False

    def test_toggles_both_ways(self):
        model = self._model()
        model.set_grad_checkpointing(True)
        assert model.grad_checkpointing is True
        model.set_grad_checkpointing(False)
        assert model.grad_checkpointing is False

    def test_checkpointing_reduces_retained_activations(self):
        """The whole point: fewer bytes held between forward and backward."""
        x = torch.randn(2, 4, 64, 64)
        timesteps = torch.tensor([0, 10])

        plain = self._model()
        plain.train()
        baseline = self._saved_bytes(plain, x, timesteps)

        ckpt = self._model()
        ckpt.train()
        ckpt.set_grad_checkpointing(True)
        reduced = self._saved_bytes(ckpt, x, timesteps)

        assert baseline > 0, "tally captured nothing — the comparison would be vacuous"
        assert reduced < baseline / 2, (
            f"checkpointing retained {reduced} bytes vs {baseline} baseline; "
            "expected at least a 2x reduction"
        )

    def test_gradients_are_unchanged(self):
        """Recompute changes WHEN activations exist, never WHAT gradients are.

        A block that was RNG-dependent or that mutated running statistics would
        recompute differently and silently corrupt training; this is the tripwire.
        """
        x = torch.randn(2, 4, 64, 64)
        timesteps = torch.tensor([0, 10])

        def grads(enable):
            torch.manual_seed(4242)
            model = self._model()
            model.train()
            if enable:
                model.set_grad_checkpointing(True)
            model(x, timesteps=timesteps).pow(2).mean().backward()
            return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

        plain, ckpt = grads(False), grads(True)

        assert plain, "no parameter received a gradient — the comparison would be vacuous"
        assert set(plain) == set(ckpt), (
            "checkpointing changed WHICH parameters receive gradients: "
            f"missing={sorted(set(plain) - set(ckpt))} extra={sorted(set(ckpt) - set(plain))}"
        )
        for name in plain:
            torch.testing.assert_close(
                plain[name], ckpt[name], rtol=1e-5, atol=1e-6, msg=f"gradient diverged for {name}"
            )

    def test_inactive_without_grad(self):
        """Under eval/no_grad, recompute would buy nothing and cost a forward.

        The 28-step reverse sampler runs there, so this is not hypothetical.
        """
        model = self._model()
        model.set_grad_checkpointing(True)

        model.eval()
        assert model._checkpointing_active() is False

        model.train()
        assert model._checkpointing_active() is True
        with torch.no_grad():
            assert model._checkpointing_active() is False


# ─────────────────────────────────────────────────────────────────────────────
# Radial band tokens: the per-up-step seam (#2117)
#
# Registering a module and having the decoder call it are different claims, so
# the firing is observed with a hook rather than read off the constructor. The
# unknown-knob refusal is planted in both shapes it can take, because this
# constructor ends in ``**kwargs`` and nothing downstream would notice.
# ─────────────────────────────────────────────────────────────────────────────

_ARMED = {"radial_band_tokens_bands": 8, "radial_band_tokens_rungs": 29}

_TOKEN_COMMON = {
    "in_channels": 8,
    "out_channels": 8,
    "features": (8, 16, 32),
    "time_embedding_dim": 32,
    "feature_domain": "kspace",
}


def _token_unet(**extra):
    import torch

    from spectramr.models.generators.complex_unet import ComplexUNet

    torch.manual_seed(7)
    return ComplexUNet(**_TOKEN_COMMON, **extra).eval()


class TestRadialBandTokens:
    """Opt-in per-annulus, per-rung conditioning at every decoder up-step."""

    def test_the_knob_is_off_by_default(self):
        assert _token_unet().radial_band_tokens is None

    def test_declaring_bands_builds_the_block(self):
        from spectramr.models.blocks.radial_band_tokens import RadialBandTokens

        assert isinstance(_token_unet(**_ARMED).radial_band_tokens, RadialBandTokens)

    def test_it_fires_once_per_up_step(self):
        """Observed, not inferred: registering a module is the easy half."""
        model = _token_unet(**_ARMED)
        fired: list[int] = []
        model.radial_band_tokens.register_forward_hook(lambda m, i, o: fired.append(int(i[1][0])))
        with torch.no_grad():
            model(_kspace_input(), torch.tensor([3]))
        assert len(fired) == len(model.ups)
        assert set(fired) == {3}, "every site must see the rung the forward was given"

    def test_an_armed_backbone_reproduces_its_control_at_initialisation(self):
        """Bit-exact, so the A/B delta is the mechanism and not a reseeded model.

        This holds only because the block is constructed LAST: a module built
        earlier consumes RNG draws and shifts every later initialisation.
        """
        x, t = _kspace_input(), torch.tensor([3])
        with torch.no_grad():
            assert torch.equal(_token_unet(**_ARMED)(x, t), _token_unet()(x, t))

    def test_a_trained_bank_changes_the_output(self):
        """Otherwise the arm is an expensive way to rerun its control."""
        model, x, t = _token_unet(**_ARMED), _kspace_input(), torch.tensor([3])
        with torch.no_grad():
            before = model(x, t)
            model.radial_band_tokens.write_heads[1].weight.normal_(0, 0.5)
            model.radial_band_tokens.bank.normal_(0, 1.0)
            assert not torch.equal(model(x, t), before)

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            ({"radial_band_tokens_bnads": 8}, "unrecognised"),
            ({"radial_band_tokens_bands": 8, "radial_band_tokens_rung": 29}, "unrecognised"),
        ],
    )
    def test_a_misspelled_knob_in_the_family_raises(self, extra, message):
        """Planted violations: ``**kwargs`` would otherwise absorb both silently."""
        with pytest.raises(ValueError, match=message):
            _token_unet(**extra)

    def test_bands_without_a_rung_count_raises(self):
        with pytest.raises(ValueError, match="without a rung count"):
            _token_unet(radial_band_tokens_bands=8)

    def test_image_domain_features_raise(self):
        """A radial annulus is not defined on an image-domain feature map."""
        from spectramr.models.generators.complex_unet import ComplexUNet

        with pytest.raises(ValueError, match="feature_domain"):
            ComplexUNet(
                **{**_TOKEN_COMMON, "feature_domain": "image"},
                **_ARMED,
            )

    def test_a_forward_without_timesteps_raises(self):
        """A bank silently defaulted to rung 0 is a global embedding, wired but inert."""
        with pytest.raises(ValueError, match="rung-indexed"):
            _token_unet(**_ARMED)(_kspace_input())

    def test_checkpointing_still_cuts_retained_activations_with_the_block(self):
        """The widened (up_block -> norm -> tokens) unit must not strand its scale
        tensors outside the checkpoint, which is what the memory budget relies on."""
        x, t = _kspace_input(), torch.tensor([3])
        plain = _token_unet(**_ARMED)
        plain.train()
        baseline = TestGradCheckpointing._saved_bytes(plain, x, t)
        checkpointed = _token_unet(**_ARMED)
        checkpointed.train()
        checkpointed.set_grad_checkpointing(True)
        assert TestGradCheckpointing._saved_bytes(checkpointed, x, t) < baseline


def _capture_t_sin(model: ComplexUNet, x: torch.Tensor, timesteps: torch.Tensor, **kwargs):
    """The sinusoidal basis actually fed to ``time_mlp`` for this forward call."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs):
        captured["t_sin"] = inputs[0].detach().clone()

    handle = model.time_mlp.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(x, timesteps, **kwargs)
    finally:
        handle.remove()
    return captured["t_sin"]


class TestTimestepNormalization:
    """Findings 15+28: the ``max_timesteps`` division must be a property of the
    sample and the declared horizon, never of what else shares its batch.

    The pre-fix code divided only when ``timesteps.max() > 1.0``, so an
    all-t=1 batch skipped the division and embedded t=1 as if it were 1.0 --
    the code for the LAST rung, not 1/max_timesteps -- while the identical
    t=1 scaled correctly the moment a larger t shared its batch.
    """

    @staticmethod
    def _model() -> ComplexUNet:
        torch.manual_seed(0)
        return ComplexUNet(
            in_channels=8,
            out_channels=8,
            features=(8, 16),
            img_size=(32, 32),
            attention_type="none",
        ).eval()

    def test_t1_scales_the_same_whether_alone_or_with_a_larger_companion(self):
        model, x = self._model(), torch.randn(2, 8, 32, 32)
        solo = _capture_t_sin(model, x, torch.tensor([1, 1]), max_timesteps=29.0)
        mixed = _capture_t_sin(model, x, torch.tensor([1, 28]), max_timesteps=29.0)
        assert torch.equal(solo[0], mixed[0])

    def test_t1_with_a_declared_horizon_encodes_1_over_29_not_1_0(self):
        model, x = self._model(), torch.randn(2, 8, 32, 32)
        got = _capture_t_sin(model, x, torch.tensor([1, 1]), max_timesteps=29.0)
        expected = model._get_sinusoidal_embedding(
            torch.tensor([1 / 29, 1 / 29]), model.time_embedding_dim
        )
        assert torch.equal(got, expected)

    def test_missing_max_timesteps_still_does_not_depend_on_batch_content(self):
        """Even the legacy no-horizon fallback must not branch on the batch."""
        model, x = self._model(), torch.randn(2, 8, 32, 32)
        solo = _capture_t_sin(model, x, torch.tensor([1, 1]))
        mixed = _capture_t_sin(model, x, torch.tensor([1, 28]))
        assert torch.equal(solo[0], mixed[0])

    def test_non_positive_max_timesteps_raises(self):
        model, x = self._model(), torch.randn(2, 8, 32, 32)
        with pytest.raises(ValueError, match="max_timesteps"):
            model(x, torch.tensor([1, 1]), max_timesteps=0.0)

    def test_missing_max_timesteps_warns_once_per_instance(self, caplog):
        model, x = self._model(), torch.randn(2, 8, 32, 32)
        with caplog.at_level("WARNING"):
            model(x, torch.tensor([1, 1]))
            model(x, torch.tensor([2, 2]))
        occurrences = sum("No max_timesteps supplied" in r.message for r in caplog.records)
        assert occurrences == 1
