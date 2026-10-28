"""Paired tests for LatentDiffusionGenerator's ``vae_backbone`` selection.

Covers the transfer-learning seam added alongside pretrained_vae_adapter.py: the
generator can build either the from-scratch custom ``AutoencoderKL`` (default) or
a frozen pretrained Stable-Diffusion VAE. ``diffusers`` is NOT required — the SD
branch is exercised only up to its clear dependency error (via monkeypatch) and
its 2D-only guard, both of which fire before any network download.
"""

from __future__ import annotations

import sys

import pytest
import torch

from spectramr.models.generators.latent_diffusion_generator import (
    AutoencoderKL,
    LatentDiffusionGenerator,
    LatentDiffusionGeneratorConfig,
)


def _config(spatial_dims: int = 2) -> LatentDiffusionGeneratorConfig:
    return LatentDiffusionGeneratorConfig(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        timesteps=5,
        spatial_dims=spatial_dims,
        device="cpu",
    )


def test_default_backbone_builds_custom_autoencoder():
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    assert gen._vae_backbone == "custom"
    assert isinstance(gen.autoencoder, AutoencoderKL)


def test_unknown_vae_backbone_raises_no_silent_fallback():
    with pytest.raises(ValueError, match="vae_backbone"):
        LatentDiffusionGenerator(_config(), num_layers=3, vae_backbone="monai")


def test_sd_backbone_requires_diffusers(monkeypatch):
    # Force `import diffusers` to fail regardless of install state.
    monkeypatch.setitem(sys.modules, "diffusers", None)
    with pytest.raises(ImportError, match="diffusers"):
        LatentDiffusionGenerator(_config(), num_layers=3, vae_backbone="sd")


def test_sd_backbone_is_2d_only():
    with pytest.raises(ValueError, match=r"2D-only|spatial_dims"):
        LatentDiffusionGenerator(_config(spatial_dims=3), vae_backbone="sd")


# ── SR3 conditioning de-facade (2026-07 ldm_two_stage triage) ──────────────────
# conditional_translation was swallowed by **kwargs → the SR3 concat was never
# built and the model ran unconditionally. These tests are the mechanism-fires
# guard that would have caught the facade.


def _conditional_kwargs(**over):
    base = {
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 4,
        "base_channels": 16,
        "timesteps": 5,
        "spatial_dims": 2,
        "conditional_translation": True,
        "conditioning_key": "concat",
        "num_layers": 3,
    }
    base.update(over)
    return base


def _first_conv_in_channels(gen) -> int:
    import torch.nn as nn

    for m in gen.denoise_net.modules():
        if isinstance(m, nn.Conv2d):
            return m.in_channels
    raise AssertionError("no Conv2d in denoise_net")


def test_conditional_translation_threaded_and_unet_doubled():
    """The knob reaches config AND the UNet first conv doubles for the concat."""
    gen = LatentDiffusionGenerator(**_conditional_kwargs())
    assert gen.config.conditional_translation is True
    assert _first_conv_in_channels(gen) == 2 * gen.latent_channels


def test_unconditional_lodm_unet_not_doubled():
    """An unconditional LDM must keep single-width UNet input (no regression)."""
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    assert gen.config.conditional_translation is False
    assert _first_conv_in_channels(gen) == gen.latent_channels


def test_forward_conditional_requires_condition_image():
    gen = LatentDiffusionGenerator(**_conditional_kwargs())
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 5, (2,))
    with pytest.raises(ValueError, match="condition_image is None"):
        gen.forward(x, t, condition_image=None)


def test_forward_conditional_runs_with_condition_image():
    gen = LatentDiffusionGenerator(**_conditional_kwargs())
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 5, (2,))
    out = gen.forward(x, t, condition_image=torch.randn(2, 1, 64, 64))
    assert tuple(out.shape) == (2, 4, 8, 8)


def test_unknown_conditioning_key_raises():
    with pytest.raises(ValueError, match="conditioning_key"):
        LatentDiffusionGenerator(**_conditional_kwargs(conditioning_key="crossattn"))


def test_contrast_embed_dim_must_match_time_emb_dim():
    with pytest.raises(ValueError, match="contrast_embed_dim"):
        LatentDiffusionGenerator(
            **_conditional_kwargs(use_contrast_guidance=True, contrast_embed_dim=64)
        )


# ── contrast guidance actually reaches the output ────────────────────────────
# The tests above prove the SR3 *concat* seam is built. Nothing proved the OTHER
# conditioning channel — per-contrast guidance — changes anything, and it is the
# harder one to check because of the trap the next test pins.


def _run(gen, contrast_idx, cond_image, *, seed=1234):
    """Forward with everything held fixed except the conditioning under test."""
    torch.manual_seed(seed)  # re-seed per call: forward() may draw internally
    with torch.no_grad():
        return gen.forward(
            torch.zeros(2, 4, 8, 8) + 0.5,
            torch.zeros(2, dtype=torch.long) + 3,
            context={"contrast_idx": contrast_idx} if contrast_idx is not None else None,
            condition_image=cond_image,
        )


def _contrast_gen():
    gen = LatentDiffusionGenerator(
        **_conditional_kwargs(
            use_contrast_guidance=True, contrast_embed_dim=128, num_contrasts=4
        )
    )
    # eval(): forward() applies 10% classifier-free-guidance dropout to the
    # condition latent while training, which would make any "outputs differ"
    # assertion below a coin flip.
    return gen.eval()


def test_contrast_guidance_is_inert_at_init_because_adagn_is_zero_init():
    """DOCUMENTS A TRAP: at init, contrast_idx provably cannot move the output.

    ``AdaptiveGroupNorm`` zero-initializes its projection
    (``normalization.py`` — ``proj.weight.data.zero_()`` / ``bias.data.zero_()``),
    so ``norm(x) * (1 + scale) + shift`` collapses to ``norm(x)`` for ANY
    embedding until those weights train. The contrast embedding travels the AdaGN
    path (``emb_decoder = emb + contrast_context``), so on a freshly built model
    switching contrasts changes the output by exactly 0.

    That is correct-by-design, NOT a facade — but it means the obvious probe
    ("build the model, flip contrast_idx, see if anything moves") reports a dead
    mechanism for a live one. This test pins the trap so the next person reads it
    before filing the bug; ``test_contrast_guidance_changes_output`` below is the
    check that actually has power.
    """
    from spectramr.models.blocks.normalization import AdaptiveGroupNorm

    gen = _contrast_gen()
    agn = [m for m in gen.modules() if isinstance(m, AdaptiveGroupNorm)]
    assert agn, "expected AdaptiveGroupNorm layers in the latent UNet"
    assert all(m.proj.weight.abs().max() == 0 for m in agn)

    cond = torch.randn(2, 1, 64, 64)
    a = _run(gen, torch.zeros(2, dtype=torch.long), cond)
    b = _run(gen, torch.full((2,), 3, dtype=torch.long), cond)
    assert torch.equal(a, b)


def test_contrast_guidance_changes_output():
    """Mechanism-fires guard: contrast_idx must move the output once AdaGN trains.

    Perturbing the AdaGN projections off their zero init is what a few optimizer
    steps do; it is the minimum state in which the contrast path can express
    itself at all. If this ever fails, the embedding is computed and discarded —
    a real pitfall #16 facade — and every contrast-conditioned arm is silently
    contrast-agnostic while still paying for the embedding.
    """
    import torch.nn as nn

    from spectramr.models.blocks.normalization import AdaptiveGroupNorm

    gen = _contrast_gen()
    torch.manual_seed(7)
    for m in gen.modules():
        if isinstance(m, AdaptiveGroupNorm):
            nn.init.normal_(m.proj.weight, std=0.02)

    cond = torch.randn(2, 1, 64, 64)
    a = _run(gen, torch.zeros(2, dtype=torch.long), cond)
    b = _run(gen, torch.full((2,), 3, dtype=torch.long), cond)
    assert not torch.allclose(a, b), "contrast_idx did not reach the output"


def test_contrast_guidance_without_contrast_idx_raises():
    """Declared guidance + absent contrast_idx must RAISE, not run unconditioned.

    ``_process_slices`` gates the contrast path on ``contrast_idx is not None``,
    so before this guard a model declaring ``use_contrast_guidance`` and never
    receiving the index trained contrast-AGNOSTICALLY while advertising
    per-contrast conditioning — carrying an ``nn.Embedding`` whose gradient is
    always zero (pitfall #16). Neither contrast audit check covered this
    direction: ``multi_contrast_model_support`` guards data->model, and
    ``contrast_conditioning_strategy_threaded`` keys off a different kwarg
    (``use_contrast_conditioning``).
    """
    gen = _contrast_gen()
    with pytest.raises(ValueError, match="contrast_idx is None"):
        gen.forward(
            torch.randn(2, 4, 8, 8),
            torch.randint(0, 5, (2,)),
            context=None,
            condition_image=torch.randn(2, 1, 64, 64),
        )


def test_contrast_guidance_guard_is_forward_only_sample_stays_permissive():
    """sample() must NOT raise on contrast_idx=None — CFG needs that path.

    Unconditional sampling is a legitimate classifier-free-guidance use, so the
    guard belongs on the training forward only. Pinning this keeps a later
    "tighten the guard everywhere" change from silently removing CFG.
    """
    gen = _contrast_gen()
    out = gen.sample(
        (1, 1, 64, 64),
        device="cpu",
        condition_image=torch.randn(1, 1, 64, 64),
        contrast_idx=None,
        num_inference_steps=1,
    )
    assert tuple(out.shape[-2:]) == (64, 64)


def test_custom_backbone_is_2d_only():
    """spatial_dims=3 must RAISE at construction, not die at the first encode.

    AutoencoderKL branches Conv2d/Conv3d for quant_conv but its encoder/decoder
    submodules are hardcoded nn.Conv2d, so a 3-D instance used to CONSTRUCT and
    then fail mid-forward with a bare "Expected 3D (unbatched) or 4D (batched)
    input to conv2d" — a shape error with nothing pointing at the config key
    that caused it. The `sd` backbone always had this guard; the custom one did
    not.
    """
    with pytest.raises(ValueError, match=r"2D-only|spatial_dims"):
        LatentDiffusionGenerator(_config(spatial_dims=3), num_layers=3)


def test_registry_declares_2d_only_for_both_names():
    """The advertised capability must match what the model can do (#8).

    ``spatial_dims=(2, 3)`` is how a 3-D arm got written against this model: the
    declaration is what the audit spec card and check_workflow_spatial_rank read.
    ``stubs.py`` aliases ``latent_diffusion`` onto the same registry entry, so
    both names are asserted here — a future split of the two registrations must
    keep them in agreement.
    """
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    for name in ("latent_gan_generator", "latent_diffusion"):
        caps = MODEL_REGISTRY[name]["capabilities"]
        assert caps.spatial_dims == (2,), f"{name} advertises {caps.spatial_dims}"


def test_sr3_condition_image_changes_output_at_init():
    """The concat path does NOT go through AdaGN, so it is live from step 0.

    This is the control for the two tests above: it shows the zero-init
    explanation is specific to the embedding path, not a property of the whole
    forward, so "no output change" on the contrast path cannot be waved away as
    the model being untrained in general.
    """
    gen = _contrast_gen()
    ci = torch.zeros(2, dtype=torch.long)
    a = _run(gen, ci, torch.randn(2, 1, 64, 64))
    b = _run(gen, ci, torch.zeros(2, 1, 64, 64))
    assert not torch.allclose(a, b), "condition_image did not reach the output"


def test_post_quant_conv_applied_once_on_round_trip():
    """encode_to_latent must NOT apply post_quant_conv (decode already does)."""
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    calls = {"n": 0}
    gen.autoencoder.post_quant_conv.register_forward_hook(
        lambda m, i, o: calls.__setitem__("n", calls["n"] + 1)
    )
    z = gen.encode_to_latent(torch.randn(1, 1, 64, 64))
    _ = gen.decode_from_latent(z)
    assert calls["n"] == 1


def test_sample_crops_to_requested_odd_size():
    """The latent floor (//ds) can undershoot; sample() aligns to the request."""
    gen = LatentDiffusionGenerator(**_conditional_kwargs())
    gen.eval()
    out = gen.sample(
        (1, 1, 72, 72),
        device="cpu",
        condition_image=torch.randn(1, 1, 72, 72),
        contrast_idx=torch.tensor([0]),
    )
    assert tuple(out.shape[-2:]) == (72, 72)


# ── latent scaling knob (2026-07): set_latent_statistics had no caller ─────────


def test_latent_scaling_factor_sets_std():
    gen = LatentDiffusionGenerator(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        timesteps=5,
        spatial_dims=2,
        latent_scaling_factor=2.45,
        num_layers=3,
    )
    assert float(gen.latent_std.flatten()[0]) == pytest.approx(1.0 / 2.45, abs=1e-5)


def test_latent_scaling_absent_is_identity_no_regression():
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    assert float(gen.latent_std.flatten()[0]) == pytest.approx(1.0)


def test_latent_scaling_factor_must_be_positive():
    with pytest.raises(ValueError, match="latent_scaling_factor"):
        LatentDiffusionGenerator(_config(), num_layers=3, latent_scaling_factor=-1.0)


def test_latent_scaling_factor_rejected_on_sd_backbone():
    pytest.importorskip("diffusers")
    with pytest.raises(ValueError, match="latent_scaling_factor"):
        LatentDiffusionGenerator(
            _config(), num_layers=3, vae_backbone="sd", latent_scaling_factor=2.0
        )


# ---------------------------------------------------------------------------
# Reverse-schedule SSOT (2026-07-11 stage-2 LDM triage).
#
# The strategy q_samples with ``training.diffusion.noise_schedule`` while the
# generator's internal reverse ``Diffusion`` was built from the dataclass
# default ``beta_schedule="linear"``, which no YAML ever wired. Training on a
# cosine forward trajectory and sampling with linear posterior coefficients
# desyncs the reverse process → the decoded image collapses to ~black
# (val_psnr≈6 dB on both stage-2 arms while train_psnr≈32).
# ---------------------------------------------------------------------------


def test_set_diffusion_schedule_rebinds_reverse_process():
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    assert gen.beta_schedule == "linear"

    gen.set_diffusion_schedule(timesteps=7, beta_schedule="cosine", device="cpu")

    assert gen.beta_schedule == "cosine"
    assert gen.timesteps == 7
    assert gen.diffusion.timesteps == 7
    # The betas must come from the cosine schedule, not the linear one.
    linear = LatentDiffusionGenerator(_config(), num_layers=3).diffusion.betas
    assert not torch.allclose(gen.diffusion.betas[: min(5, 7)], linear[:5])


def test_set_diffusion_schedule_rejects_unknown_schedule():
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    with pytest.raises(ValueError, match="beta schedule"):
        gen.set_diffusion_schedule(timesteps=5, beta_schedule="not_a_schedule")


def test_beta_schedule_kwarg_is_honoured():
    """model_kwargs.beta_schedule must reach the reverse process (#15).

    Exercises the production path: the model builder passes model_kwargs, NOT a
    config object (an explicit ``config=`` makes the scalar args a documented
    no-op).
    """
    gen = LatentDiffusionGenerator(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        timesteps=5,
        beta_schedule="cosine",
        device="cpu",
        num_layers=3,
    )
    assert gen.beta_schedule == "cosine"
    assert gen.diffusion.betas.shape[0] == 5


# ---------------------------------------------------------------------------
# Stage-1 VAE hand-off.
#
# The training builder reads the stage-1 RunState's generator part and hands it to
# ``load_autoencoder_state``; the generator does no checkpoint IO itself. Before the
# 2026-07-11 fix the loader missed the writer's key and the LDM silently trained
# against a randomly-initialised VAE, so every incomplete restore must raise.
# ---------------------------------------------------------------------------


def _stage1_state():
    """The generator part a stage-1 ``autoencoder_kl_pretrain`` run writes."""
    from spectramr.models.generators.autoencoder_pretrain_generator import (
        AutoencoderPretrainGenerator,
    )

    # Same autoencoder geometry the LDM builds from ``_config()`` so the
    # state-dict keys AND shapes line up (base_channels is the only knob the
    # LDM forwards; num_layers/downsample_factor take AutoencoderKL defaults).
    stage1 = AutoencoderPretrainGenerator(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        spatial_dims=2,
    )
    return dict(stage1.state_dict()), stage1


def test_load_autoencoder_state_strips_the_stage1_prefix():
    """The stage-1 generator's keys sit under ``autoencoder.``; they must land."""
    state, stage1 = _stage1_state()
    assert any(k.startswith("autoencoder.") for k in state)

    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    gen.load_autoencoder_state(state, freeze=True, source="stage1/best.safetensors")

    # Weights actually transferred (not left at random init).
    ref = stage1.autoencoder.state_dict()
    for name, param in gen.autoencoder.state_dict().items():
        if name in ref and param.is_floating_point():
            assert torch.allclose(param, ref[name]), f"{name} did not load"
            break
    else:  # pragma: no cover - defensive
        pytest.fail("no comparable autoencoder parameter found")

    # freeze=True must actually freeze.
    assert all(not p.requires_grad for p in gen.autoencoder.parameters())


def test_the_constructor_no_longer_reads_the_file():
    """The path is the builder's to read; the constructor only validates the backbone."""
    LatentDiffusionGenerator(
        _config(), num_layers=3, vae_backbone="custom", vae_checkpoint_path="/does/not/exist"
    )


def test_vae_checkpoint_with_no_matching_keys_raises():
    """Architecture divergence between stage-1 and stage-2 must RAISE (#9)."""
    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    with pytest.raises(RuntimeError, match="contained no keys matching"):
        gen.load_autoencoder_state(
            {"totally.unrelated.weight": torch.zeros(2)}, freeze=True, source="x.pt"
        )


def test_partially_matching_vae_checkpoint_raises():
    """A state that restores only SOME autoencoder tensors must RAISE (#9).

    ``load_state_dict(strict=False)`` tolerates missing keys by leaving them at
    their random init, and the emptiness guard only catches a *total* miss. A
    stage-1 VAE whose key set merely diverges (different depth) would otherwise
    load a PARTIALLY random autoencoder and still report PASS.
    """
    state, _ = _stage1_state()
    # Key names and shapes still line up; the restore is incomplete. This is the
    # depth-drift case (shape drift already raises inside load_state_dict).
    del state[next(k for k in state if k.startswith("autoencoder."))]

    gen = LatentDiffusionGenerator(_config(), num_layers=3)
    with pytest.raises(RuntimeError, match="restored only"):
        gen.load_autoencoder_state(state, freeze=True, source="x.pt")


class TestValidationSamplerSteps:
    """``sample(num_inference_steps=...)`` — wiring for ``validation.sampler_steps``.

    ``DiffusionTrainingStrategy._generate_validation_prediction`` forwards the
    resolved step count only under a parameter name the sampler's signature
    actually exposes. Before this seam existed, ``sampler_steps: 25`` on the
    ldm_ulf_to_hf stage-2 arms was silently dropped and every validation ran the
    full 1000-step reverse loop (40x the declared cost).
    """

    def _gen(self, timesteps: int = 20) -> LatentDiffusionGenerator:
        cfg = LatentDiffusionGeneratorConfig(
            in_channels=1,
            out_channels=1,
            latent_channels=4,
            base_channels=16,
            timesteps=timesteps,
            spatial_dims=2,
            device="cpu",
        )
        return LatentDiffusionGenerator(cfg, num_layers=3)

    def _count_denoiser_calls(self, gen, monkeypatch) -> list[int]:
        seen: list[int] = []

        def _spy(x, t, mode, **kwargs):
            seen.append(int(t.flatten()[0].item()))
            return torch.zeros_like(x[:, : gen.latent_channels])

        monkeypatch.setattr(gen, "_process_slices", _spy)
        return seen

    def test_strategy_probe_finds_a_step_parameter(self):
        # Bind to the strategy's REAL probe list, not a copy: at least one of
        # those names must be on sample() or the configured sampler_steps is
        # dropped and validation silently runs the full chain.
        import inspect

        from spectramr.infrastructure.training.strategies.diffusion import (
            _SAMPLER_STEP_PARAM_NAMES,
        )

        params = inspect.signature(LatentDiffusionGenerator.sample).parameters
        assert any(name in params for name in _SAMPLER_STEP_PARAM_NAMES)

    def test_num_inference_steps_limits_denoiser_evaluations(self, monkeypatch):
        gen = self._gen(timesteps=20)
        seen = self._count_denoiser_calls(gen, monkeypatch)

        gen.sample((1, 1, 16, 16), device="cpu", num_inference_steps=5)

        assert len(seen) == 5
        # Strided schedule must run strictly high -> low and end at t=0.
        assert seen == sorted(seen, reverse=True)
        assert seen[-1] == 0

    def test_the_full_chain_passes_the_timestep_to_p_sample_step(self, monkeypatch):
        """PLANTED (#482): the loop counter went in, so the FIRST step (t = T-1) dropped
        its posterior noise and the last one kept it."""
        gen = self._gen(timesteps=20)
        self._count_denoiser_calls(gen, monkeypatch)
        passed: list[int] = []
        real = gen.diffusion.p_sample_step

        def _spy(x, t, noise, t_index):
            passed.append(int(t_index))
            return real(x, t, noise, t_index)

        monkeypatch.setattr(gen.diffusion, "p_sample_step", _spy)
        gen.sample((1, 1, 16, 16), device="cpu")
        assert passed == list(range(19, -1, -1))

    def test_default_runs_the_full_reverse_loop(self, monkeypatch):
        gen = self._gen(timesteps=20)
        seen = self._count_denoiser_calls(gen, monkeypatch)

        gen.sample((1, 1, 16, 16), device="cpu")

        assert len(seen) == 20

    def test_step_count_above_schedule_is_clamped(self, monkeypatch):
        gen = self._gen(timesteps=20)
        seen = self._count_denoiser_calls(gen, monkeypatch)

        gen.sample((1, 1, 16, 16), device="cpu", num_inference_steps=999)

        assert len(seen) == 20

    def test_single_step_starts_at_max_noise_not_zero(self, monkeypatch):
        # torch.linspace(a, b, 1) returns [a]. Spanning low->high would put the
        # lone evaluation at t=0 — telling the denoiser that pure noise is
        # already clean. sampler_steps: 1 is legal config (schema ge=1).
        gen = self._gen(timesteps=20)
        seen = self._count_denoiser_calls(gen, monkeypatch)

        gen.sample((1, 1, 16, 16), device="cpu", num_inference_steps=1)

        assert seen == [19]

    def test_strided_schedule_is_distinct_and_spans_the_chain(self, monkeypatch):
        # Dedup must never silently shorten the chain: asking for k steps has to
        # cost exactly k denoiser evaluations, from max noise down to t=0.
        for k in (2, 3, 7, 19):
            gen = self._gen(timesteps=20)
            seen = self._count_denoiser_calls(gen, monkeypatch)
            gen.sample((1, 1, 16, 16), device="cpu", num_inference_steps=k)
            assert len(seen) == k, f"k={k} produced {len(seen)} evaluations"
            assert seen[0] == 19 and seen[-1] == 0, f"k={k} spans {seen[0]}..{seen[-1]}"


# --------------------------------------------------------------------------
# quant_conv is UNTRAINED on the from-scratch stage-1 path (2026-07 ldm review)
#
# Stage 1 trains through AutoencoderPretrainGenerator.forward -> AutoencoderKL
# .encode()/.decode(); encode() never touches quant_conv, so it keeps its random
# init in the stage-1 checkpoint. encode_to_latent used to apply it anyway, and
# decode_from_latent never inverted it -- a random 1x1 channel mix wedged
# between the trained encoder and the trained decoder.
# --------------------------------------------------------------------------


def test_encode_to_latent_matches_the_encode_stage1_trained() -> None:
    """The stage-2 latent must be the stage-1 encoder's mean, up to standardisation."""
    import torch

    from spectramr.models.generators.latent_diffusion_generator import (
        LatentDiffusionGenerator,
    )

    gen = LatentDiffusionGenerator(
        in_channels=1, out_channels=1, latent_channels=4, base_channels=8, spatial_dims=2
    )
    gen.eval()
    # Make quant_conv conspicuously non-identity, as a random init would be.
    with torch.no_grad():
        gen.autoencoder.quant_conv.weight.normal_(0.0, 1.0)
        gen.autoencoder.quant_conv.bias.normal_(0.0, 1.0)

    x = torch.randn(2, 1, 32, 32)
    z = gen.encode_to_latent(x)

    with torch.no_grad():
        mean, _ = gen.autoencoder.encode(x)
        expected = (mean - gen.latent_mean) / (gen.latent_std + 1e-8)

    assert z.shape == expected.shape
    torch.testing.assert_close(z, expected)


def test_quant_conv_receives_no_gradient_from_the_stage1_objective() -> None:
    """Pins WHY the fix is required rather than just the fix."""
    import torch

    from spectramr.models.generators.autoencoder_pretrain_generator import (
        AutoencoderPretrainGenerator,
    )

    gen = AutoencoderPretrainGenerator(
        in_channels=1, out_channels=1, latent_channels=4, base_channels=8
    )
    out = gen(torch.randn(2, 1, 32, 32))
    out["reconstruction"].square().mean().backward()

    assert gen.autoencoder.quant_conv.weight.grad is None, (
        "quant_conv now takes gradient in stage 1 — if the stage-1 objective was "
        "changed to train it, encode_to_latent must apply it again."
    )
    # The decoder-side twin IS trained, because decode() applies it.
    assert gen.autoencoder.post_quant_conv.weight.grad is not None


# ── multi-axis CFG + channel_multipliers (LM-19) ─────────────────────────────


def _cfg_gen(**kwargs) -> LatentDiffusionGenerator:
    # Explicit arguments: a config object makes the generator ignore these kwargs.
    return LatentDiffusionGenerator(
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        base_channels=16,
        timesteps=5,
        spatial_dims=2,
        num_layers=3,
        conditional_translation=True,
        use_contrast_guidance=True,
        num_contrasts=3,
        **kwargs,
    )


def test_channel_multipliers_set_the_unet_depth():
    gen = _cfg_gen(channel_multipliers=[1, 2, 4, 4])
    assert tuple(gen.denoise_net.channel_mult) == (1, 2, 4, 4)
    assert tuple(_cfg_gen().denoise_net.channel_mult) == (1, 2, 4)


@pytest.mark.parametrize("bad", [[], [1, 0], [1, "2"], "124", 4])
def test_bad_channel_multipliers_raise(bad):
    with pytest.raises(ValueError, match="channel_multipliers"):
        _cfg_gen(channel_multipliers=bad)


def test_declaring_a_no_data_axis_raises_at_construction():
    with pytest.raises(ValueError, match="thickness"):
        _cfg_gen(cfg_drop_probs={"contrast": 0.1, "thickness": 0.05})


def _wake_zero_init(gen: LatentDiffusionGenerator) -> None:
    """Zero-initialised embedding projections hide conditioning at init; perturb them."""
    with torch.no_grad():
        for p in gen.denoise_net.parameters():
            if not p.any():
                p.normal_(0.0, 0.1)


def test_contrast_drop_changes_the_training_forward():
    """A dropped contrast row equals the contrast_idx=None path, not index 0."""
    torch.manual_seed(0)
    gen = _cfg_gen(cfg_drop_probs={"condition": 0.0, "contrast": 0.999999}).train()
    gen.denoise_net.eval()  # U-Net dropout off; the generator stays in training mode
    _wake_zero_init(gen)
    z = torch.randn(2, 4, 4, 4)
    t = torch.tensor([1, 2])
    cond = torch.randn(2, 1, 32, 32)
    idx = torch.tensor([0, 2])
    dropped = gen(z, t, context={"contrast_idx": idx}, condition_image=cond)
    unet_in = torch.cat([z, gen.encode_condition(cond)], dim=1)
    with torch.no_grad():
        null = gen._process_slices(unet_in, t, contrast_idx=None)
        kept = gen._process_slices(unet_in, t, contrast_idx=idx)
    assert torch.allclose(dropped, null, atol=1e-5)
    assert not torch.allclose(dropped, kept)


def test_absent_cfg_knobs_reproduce_the_legacy_condition_drop():
    """Same seed, same 10% bernoulli mask on the ULF latent as before LM-19."""
    gen = _cfg_gen().train()
    gen.denoise_net.eval()
    z = torch.randn(4, 4, 4, 4)
    t = torch.tensor([1, 2, 3, 4])
    cond = torch.randn(4, 1, 32, 32)
    idx = torch.tensor([0, 1, 2, 0])
    torch.manual_seed(3)
    out = gen(z, t, context={"contrast_idx": idx}, condition_image=cond)
    torch.manual_seed(3)
    mask = torch.bernoulli(torch.zeros(4) + 0.9).view(-1, 1, 1, 1)
    with torch.no_grad():
        ref = gen._process_slices(
            torch.cat([z, gen.encode_condition(cond) * mask], dim=1), t, contrast_idx=idx
        )
    assert torch.allclose(out, ref, atol=1e-6)


def test_guided_sampling_changes_the_output_and_batches_the_passes(monkeypatch):
    torch.manual_seed(0)
    gen = _cfg_gen(cfg_drop_probs={"condition": 0.1, "contrast": 0.1}, cfg_weights={"condition": 2.0, "contrast": 2.0}).eval()
    _wake_zero_init(gen)
    plain = _cfg_gen().eval()
    plain.load_state_dict(gen.state_dict())
    seen = []
    orig = gen.denoise_net.forward
    monkeypatch.setattr(gen.denoise_net, "forward", lambda x, *a, **k: seen.append(x.shape[0]) or orig(x, *a, **k))
    kw = {
        "shape": (1, 1, 32, 32),
        "condition_image": torch.randn(1, 1, 32, 32),
        "contrast_idx": torch.tensor([1]),
        "num_inference_steps": 2,
    }
    torch.manual_seed(1)
    guided = gen.sample(**kw)
    torch.manual_seed(1)
    unguided = plain.sample(**kw)
    assert not torch.allclose(guided, unguided)
    assert seen == [3, 3], "null + two axes in one U-Net call per step"


# ── Pattern-C acquisition FiLM (LM-20) ───────────────────────────────────────

_LEIDEN_MAP = {"T1w": 0, "T2w": 1, "FLAIR": 2}
_LEIDEN_ROWS = [
    {"name": "T1w", "TE": 12.0, "TR": 600.0, "TI": -1.0, "FA": 70.0},
    {"name": "T2w", "TE": 80.0, "TR": 4000.0, "TI": -1.0, "FA": 90.0},
    {"name": "FLAIR", "TE": 120.0, "TR": 8000.0, "TI": 2200.0, "FA": 90.0},
]


def test_acquisition_film_needs_contrast_guidance():
    with pytest.raises(ValueError, match="use_contrast_guidance"):
        LatentDiffusionGenerator(
            in_channels=1, out_channels=1, latent_channels=4, base_channels=16,
            timesteps=5, num_layers=3, acquisition_film=True,
        )


def test_acquisition_film_must_be_a_bool():
    with pytest.raises(ValueError, match="true/false"):
        _cfg_gen(acquisition_film="yes")


def test_acquisition_film_off_adds_no_parameters():
    assert not any(k.startswith("acquisition_film") for k in _cfg_gen().state_dict())


def test_acquisition_film_changes_the_forward_and_the_sampler():
    torch.manual_seed(0)
    on = _cfg_gen(acquisition_film=True).eval()
    on.set_acquisition_params(_LEIDEN_MAP, _LEIDEN_ROWS)
    _wake_zero_init(on)
    torch.nn.init.normal_(on.acquisition_film.to_gamma_beta.weight, std=0.5)
    off = _cfg_gen().eval()
    off.load_state_dict(on.state_dict(), strict=False)
    z, t = torch.randn(2, 4, 4, 4), torch.tensor([1, 2])
    cond, ctx = torch.randn(2, 1, 32, 32), {"contrast_idx": torch.tensor([0, 2])}
    assert not torch.allclose(on(z, t, context=ctx, condition_image=cond), off(z, t, context=ctx, condition_image=cond))
    kw = {
        "shape": (1, 1, 32, 32),
        "condition_image": torch.randn(1, 1, 32, 32),
        "contrast_idx": torch.tensor([2]),
        "num_inference_steps": 2,
    }
    torch.manual_seed(1)
    a = on.sample(**kw)
    torch.manual_seed(1)
    b = off.sample(**kw)
    assert not torch.allclose(a, b)


def test_acquisition_film_unbound_raises_at_forward():
    gen = _cfg_gen(acquisition_film=True).eval()
    with pytest.raises(RuntimeError, match="no acquisition table"):
        gen(torch.randn(1, 4, 4, 4), torch.tensor([1]), context={"contrast_idx": torch.tensor([0])}, condition_image=torch.randn(1, 1, 32, 32))


# ── latent_scaling_factor: auto (LM-14): calibrate std(mu), not the posterior sigma ──


def _auto_gen() -> LatentDiffusionGenerator:
    return LatentDiffusionGenerator(_config(), num_layers=3, latent_scaling_factor="auto")


def test_auto_scale_standardises_the_first_training_batch():
    torch.manual_seed(0)
    gen = _auto_gen().train()
    x = torch.rand(4, 1, 32, 32) * 2 - 1
    z = gen.encode_to_latent(x)
    assert float(z.std(unbiased=False)) == pytest.approx(1.0, rel=1e-3)
    assert bool(gen.latent_scale_calibrated)


def test_auto_scale_changes_the_latent_relative_to_the_identity():
    torch.manual_seed(0)
    x = torch.rand(4, 1, 32, 32) * 2 - 1
    gen = _auto_gen().train()
    plain = LatentDiffusionGenerator(_config(), num_layers=3)
    plain.load_state_dict(gen.state_dict(), strict=False)
    assert not torch.allclose(gen.encode_to_latent(x), plain.encode_to_latent(x))


def test_auto_scale_is_measured_once_then_frozen():
    torch.manual_seed(0)
    gen = _auto_gen().train()
    gen.encode_to_latent(torch.rand(4, 1, 32, 32))
    first = gen.latent_std.clone()
    gen.encode_to_latent(torch.rand(4, 1, 32, 32) * 10)
    assert torch.equal(gen.latent_std, first)


def test_auto_scale_survives_a_checkpoint_round_trip_in_eval():
    torch.manual_seed(0)
    trained = _auto_gen().train()
    trained.encode_to_latent(torch.rand(4, 1, 32, 32))
    restored = _auto_gen().eval()
    restored.load_state_dict(trained.state_dict())
    x = torch.rand(2, 1, 32, 32)
    assert torch.allclose(restored.encode_to_latent(x), trained.eval().encode_to_latent(x))


def test_auto_scale_uncalibrated_in_eval_raises():
    with pytest.raises(RuntimeError, match="not calibrated"):
        _auto_gen().eval().encode_to_latent(torch.rand(1, 1, 32, 32))


def test_numeric_scale_has_no_calibration_buffer():
    """Old checkpoints of numeric-scale arms must still load strictly."""
    gen = LatentDiffusionGenerator(_config(), num_layers=3, latent_scaling_factor=2.45)
    assert "latent_scale_calibrated" not in gen.state_dict()
