"""Tests for the three baseline adapter skeletons (CDiffMR, Shen 2024, FDB).

Locks Phases B / C / D of
``TODO/backlog_baseline_replication_experiment_11.md``: each adapter
declares the contract attributes the registry-dispatcher and the
provenance manifest consume, and ``forward`` raises a clear
NotImplementedError until the upstream is vendored.
"""

from __future__ import annotations

import sys

import pytest
import torch

from spectramr.infrastructure.reporting.metadata import baseline_provenance
from spectramr.models.baselines import BaselineAdapter, CoilHandling, FFTNorm
from spectramr.models.baselines._base import UpstreamLossFamily
from spectramr.models.baselines.cdiffmr import (
    _UPSTREAM_NETWORK_FILE,
    CDiffMRBaseline,
)
from spectramr.models.baselines.fdb import _FDB_SCRIPT_UTIL, FDBBaseline
from spectramr.models.baselines.shen2024 import Shen2024Baseline

# ---------------------------------------------------------------------------
# Vendored-upstream gates
# ---------------------------------------------------------------------------
#
# CDiffMR and FDB are git SUBMODULES (.gitmodules). A checkout that never ran
# `git submodule update --init` has the gitlink but no files, so every test
# below that touches upstream code fails on a missing import rather than
# skipping. That is what happened on cluster job 8004252: 9 red tests that said
# nothing about this repo.
#
# The gate reads the same SENTINEL FILE each adapter's own guard reads, so the
# skip cannot drift from the thing being guarded. Checking the DIRECTORY would
# not work: an uninitialised submodule leaves an empty directory that exists.
_VENDOR_HINT = (
    "vendored upstream absent — run `git submodule update --init --recursive`"
)
needs_cdiffmr = pytest.mark.skipif(
    not _UPSTREAM_NETWORK_FILE.exists(),
    reason=f"CDiffMR {_VENDOR_HINT}",
)
needs_fdb = pytest.mark.skipif(
    not _FDB_SCRIPT_UTIL.exists(),
    reason=f"FDB {_VENDOR_HINT}",
)


def _cuda_actually_usable() -> bool:
    """``torch.cuda.is_available()`` is True but non-functional on this box.

    Thor (sm_110) is visible to a cu126-built torch but has no compatible kernel
    image, so `is_available()` alone would let a "real loop" test attempt a genuine
    kernel launch here and fail on an unrelated hardware mismatch, not a finding.
    """
    if not torch.cuda.is_available():
        return False
    try:
        (torch.zeros(1, device="cuda") + 1).item()
        return True
    except Exception:
        return False


@pytest.mark.parametrize(
    "cls,expected_repo,expected_paper,expected_mask",
    [
        # PAPER_REF is what `baseline_provenance` stamps into a run summary, so a
        # wrong one travels into any results table built from it. Two of these
        # three were wrong and this parametrisation is what pinned them:
        #   Karaoglu2024:FDB    -> no author of the paper, and not its year. It is
        #                          Mirza et al., arXiv:2308.01096 (2023).
        #   Shen2024:CondDiffMRI -> names a conditional-diffusion method; the paper
        #                          is k-space cold diffusion (Sci. Rep. 14, 2024).
        # PREFERRED_MASK_TYPE read `gaussian_density` for both, which names neither
        # a registered accelerator nor a MaskType. The values below are each
        # author's own: CDiffMR's option file selects `fMRI_Ran_AF4_CF0.08_PE320`
        # (the random family), Shen's `sample_mask.py` builds a 2D Gaussian (#2087).
        (CDiffMRBaseline, "cdiffmr", "Huang2023:CDiffMR", "random_cartesian"),
        (
            Shen2024Baseline,
            "shen2024",
            "Shen2024:KSpaceColdDiffusion",
            "variable_density_2d_gaussian",
        ),
        (FDBBaseline, "fdb", "Mirza2023:FDB", "cartesian_peripheral"),
    ],
)
def test_adapter_declares_required_attributes(
    cls: type,
    expected_repo: str,
    expected_paper: str,
    expected_mask: str,
) -> None:
    """Each adapter sets the three required overrides."""
    assert cls.REPO_NAME == expected_repo
    assert cls.PAPER_REF == expected_paper
    assert cls.PREFERRED_MASK_TYPE == expected_mask


@pytest.mark.parametrize(
    "cls", [CDiffMRBaseline, Shen2024Baseline, FDBBaseline],
)
def test_adapter_is_baseline_adapter_subclass(cls: type) -> None:
    """Each adapter subclasses the canonical base."""
    assert issubclass(cls, BaselineAdapter)


def test_validation_sample_default_raises_never_silently_forwards() -> None:
    """FINDING 21: the base hook's default is a loud refusal, not a `forward()` fallback.

    An adapter that overrides neither `validation_sample` nor `forward` cannot exist
    (`forward` is abstract), but one that overrides only `forward` inherits this
    default -- and must NOT be able to report a single-step forward as its
    reconstruction metric by omission.
    """

    class _Minimal(BaselineAdapter):
        REPO_NAME = "minimal"
        PAPER_REF = "Nobody2026:Minimal"
        PREFERRED_MASK_TYPE = "random_cartesian"
        UPSTREAM_LOSS_FAMILY = UpstreamLossFamily.L1

        def training_loss(self, x_0, batch=None):
            return x_0.abs().mean()

        def forward(self, x, *args, **kwargs):
            return x

    adapter = _Minimal()
    x = torch.zeros(1, 2, 4, 4)
    with pytest.raises(NotImplementedError, match="no wired reverse-sampling validation"):
        adapter.validation_sample(x, x)


@pytest.mark.parametrize(
    "cls,expected_coil",
    [
        (CDiffMRBaseline, CoilHandling.RSS),
        (Shen2024Baseline, CoilHandling.RSS),
        (FDBBaseline, CoilHandling.MULTI_COIL_KSPACE),
    ],
)
def test_coil_handling_matches_paper(cls: type, expected_coil: CoilHandling) -> None:
    """Coil handling matches the paper's evaluation protocol."""
    assert cls.COIL_HANDLING is expected_coil


@pytest.mark.parametrize(
    "cls", [CDiffMRBaseline, Shen2024Baseline, FDBBaseline],
)
def test_fft_norm_is_ortho(cls: type) -> None:
    """All three baselines use the repo's canonical ortho FFT."""
    assert cls.PREFERRED_FFT_NORM is FFTNorm.ORTHO


def test_shen2024_forwards_through_the_authors_unet() -> None:
    """Shen is wired now; this pins WHAT it is wired to.

    Until 2026-09-15 the adapter raised and held zero parameters, and two tests here
    pinned that. Both went red when the network landed, which is what they were for.
    Replacing them rather than deleting them keeps the same question asked: an arm
    naming this baseline must get the authors' ``Unet``, not a stand-in.
    """
    adapter = Shen2024Baseline(image_size=64, timesteps=8)
    assert type(adapter.upstream).__name__ == "Unet"
    assert type(adapter.upstream).__module__.endswith("u_net_diffusion")

    out = adapter(torch.randn(1, 2, 64, 64))
    assert out.shape == (1, 2, 64, 64)


def test_shen2024_holds_the_published_capacity() -> None:
    """``Unet(dim=64, dim_mults=(1,2,4,8), channels=2)`` -- the notebook's build.

    A parameter count of zero was the old fact; a count that drifts from the paper's
    architecture is the new one worth catching, so the shape of the check moves with
    the code rather than being dropped.
    """
    adapter = Shen2024Baseline(image_size=64, timesteps=8)
    params = sum(p.numel() for p in adapter.parameters())
    assert params > 0, "the adapter holds no network"
    assert adapter.upstream.channels == 2, (
        "the paper builds a 2-channel network and loops it over coils; a wider one "
        "is a different architecture under the same name"
    )


def test_shen2024_sample_raises_off_an_accelerator() -> None:
    """FINDING 21/upstream `.cuda()` hard-code: the reverse loop refuses on CPU, loudly.

    ``KspaceDiffusion.sample`` hard-codes ``.cuda()`` on its per-step timestep tensor,
    so this adapter's own ``sample()`` guards it first rather than let a bare unrelated
    CUDA error surface with no context.
    """
    adapter = Shen2024Baseline(image_size=32, timesteps=5)
    x = torch.randn(1, 2, 32, 32)
    with pytest.raises(RuntimeError, match=r"hard-codes `\.cuda\(\)`"):
        adapter.sample(x)


def test_shen2024_validation_sample_delegates_to_sample() -> None:
    """`validation_sample` is `sample()` driven from the clean target -- same guard."""
    adapter = Shen2024Baseline(image_size=32, timesteps=5)
    x = torch.randn(1, 2, 32, 32)
    with pytest.raises(RuntimeError, match=r"hard-codes `\.cuda\(\)`"):
        adapter.validation_sample(input_batch=torch.zeros_like(x), target_batch=x)


@pytest.mark.skipif(not _cuda_actually_usable(), reason="no usable CUDA device on this box")
def test_shen2024_sample_runs_the_real_reverse_loop_on_an_accelerator() -> None:
    """On a working accelerator, `sample()` runs upstream's real reverse loop."""
    adapter = Shen2024Baseline(image_size=32, timesteps=3).cuda()
    x = torch.randn(1, 2, 32, 32, device="cuda")
    out = adapter.sample(x)
    assert out.shape == x.shape


def test_cdiffmr_default_opt_is_the_upstreams_published_netg_block() -> None:
    """The architecture defaults must match the upstream option file, not drift from it.

    Reads the file rather than restating numbers from it: the values were first
    transcribed by eye from a ``grep`` and the block was named ``netG``, which does
    not exist — it is ``denoise_fn``. Parsing is what caught that.

    ``_default_opt``'s docstring claimed to mirror
    ``options/CDiffMR/FastMRI/ksu/train_CDiffMR_...json`` and diverged on four of
    its five architecture fields, building a 6.08M-parameter network where the
    published one is 34.43M -- under the paper's name.

    ``attn_resolutions`` is pinned to the absolute ``[16]`` rather than a fraction
    of the input: it was ``[resolution // 4]``, which drifts with the patch size,
    so the same arm at 256 and at 320 attended at different scales.
    """
    import inspect
    import json
    import re

    from spectramr.models.baselines.cdiffmr import (
        _CDIFFMR_DIR,
        _UPSTREAM_OPTION_FILE,
        _default_opt,
    )

    option_path = _CDIFFMR_DIR / _UPSTREAM_OPTION_FILE
    if not option_path.exists():
        pytest.skip(f"CDiffMR {_VENDOR_HINT}")

    # The upstream ships JSON with `//` comments, which json.loads rejects. Strip
    # them rather than hand-copying values out: reading the file is what makes this
    # a proof of the docstring's claim instead of a restatement of it, and it goes
    # red if upstream ever moves the block.
    raw = re.sub(r"//[^\n]*", "", option_path.read_text())
    upstream = json.loads(raw)
    netg = upstream["denoise_fn"]

    opt = _default_opt(256, 2, 2)
    for field in ("emb_channels", "emb_channels_multi", "num_res_blocks", "dropout"):
        assert opt[field] == netg[field], (
            f"_default_opt[{field!r}] is {opt[field]!r} but the upstream's published "
            f"denoise_fn block says {netg[field]!r}"
        )
    assert opt["attn_resolutions"] == netg["attn_resolutions"] == [16]
    # Absolute, not derived: it was `[resolution // 4]`, so the same arm attended at
    # a different scale for every patch size. The published value must not move.
    assert _default_opt(320, 2, 2)["attn_resolutions"] == [16]

    # The step count lives in the `diffusion` block, and the adapter default must
    # agree with it. It was 50 against the published 100.
    assert inspect.signature(CDiffMRBaseline.__init__).parameters["num_steps"].default == (
        upstream["diffusion"]["time_step"]
    )


def test_fdb_default_kwargs_match_the_upstreams_published_train_command() -> None:
    """FDB's defaults must be the paper's, not the upstream library's.

    The upstream README's single-coil train command is the authority:
    ``--num_channels 128 --num_res_blocks 3 --dropout 0.3 --diffusion_steps 1000
    --undersampling_rate 2``. The adapter carried ``num_res_blocks 2`` and
    ``dropout 0.0`` -- correct for ``model_and_diffusion_defaults()`` and wrong
    for the published model, a 124.05M network against 164.30M.

    ``undersampling_rate`` is the TRAIN rate. The paper infers at R=4
    (``sample.py --R 4``); that is a different knob and neither is 4 here.
    """
    from spectramr.models.baselines.fdb import _default_model_kwargs

    kw = _default_model_kwargs(256)
    assert kw["num_channels"] == 128
    assert kw["num_res_blocks"] == 3
    assert kw["dropout"] == pytest.approx(0.3)
    assert kw["diffusion_steps"] == 1000
    assert kw["undersampling_rate"] == 2
    assert kw["learn_sigma"] is False


@needs_cdiffmr
@pytest.mark.parametrize("resolution", [256, 320])
def test_cdiffmr_attn_resolutions_16_matches_no_level_at_either_size(
    resolution: int,
) -> None:
    """FINDING 24: ``[16]`` builds no per-level ``AttnBlock`` -- upstream's own gap.

    ``_default_opt``'s docstring used to claim ``[16]`` "resolves to a real level at
    both ... verified by construction at both"; measured, it resolves to none. With
    ``emb_channels_multi: [1, 2, 2, 2]`` the four levels visit 256/128/64/32 (or
    320/160/80/40), so only the unconditional ``mid.attn_1`` is built -- one
    ``AttnBlock``, not the per-level set the paper's figure suggests. The published
    option file declares the same pair, so the network this adapter builds is
    faithful to upstream; it is upstream whose own attention never engages.
    """
    adapter = CDiffMRBaseline(resolution=resolution, num_steps=2)
    n_attn = sum(1 for m in adapter.modules() if type(m).__name__ == "AttnBlock")
    assert n_attn == 1, (
        f"expected exactly the unconditional mid.attn_1 at resolution={resolution}, "
        f"found {n_attn} AttnBlock instances"
    )


@needs_cdiffmr
@needs_fdb
@pytest.mark.parametrize(
    "cls,resolution,in_ch",
    [
        (CDiffMRBaseline, 32, 1),
        (FDBBaseline, 64, 1),
    ],
)
def test_wired_adapter_complex_round_trip(
    cls: type, resolution: int, in_ch: int
) -> None:
    """CDiffMR and FDB run end-to-end on complex MRI input.

    The adapter coerces complex ``[B, C, H, W]`` to real ``[B, 2C, H, W]``,
    forwards through the upstream UNet, and coerces back. Shape and dtype
    must round-trip exactly — anything else means the channel-coercion shim
    drifted.
    """
    kwargs = {"in_channels": 2 * in_ch, "out_channels": 2 * in_ch}
    if cls is CDiffMRBaseline:
        kwargs["resolution"] = resolution
    else:
        kwargs["image_size"] = resolution
        kwargs["bridge_steps"] = 10
    adapter = cls(**kwargs)
    x = torch.randn(1, in_ch, resolution, resolution, dtype=torch.complex64)
    y = adapter(x)
    assert y.shape == x.shape
    assert torch.is_complex(y)


@needs_cdiffmr
@needs_fdb
@pytest.mark.parametrize(
    "cls,resolution",
    [(CDiffMRBaseline, 32), (FDBBaseline, 64)],
)
def test_wired_adapter_accepts_real_input(cls: type, resolution: int) -> None:
    """Real-valued input (upstream native layout) passes through unmolested."""
    if cls is CDiffMRBaseline:
        adapter = cls(resolution=resolution)
    else:
        adapter = cls(image_size=resolution, bridge_steps=10)
    x = torch.randn(1, 2, resolution, resolution)
    y = adapter(x)
    assert y.shape == x.shape
    assert not torch.is_complex(y)


# ---------------------------------------------------------------------------
# Finding 20: the lazily-built upstream process object must not register as
# an `nn.Module` child of the adapter -- it wraps `self.upstream` as
# `denoise_fn`, so a plain attribute assignment would double every parameter
# into `state_dict()` and fail the strict reload both checkpoint readers use.
# ---------------------------------------------------------------------------


@needs_cdiffmr
def test_cdiffmr_upstream_process_does_not_double_the_checkpoint() -> None:
    """Building the lazy `GaussianDiffusion` must not change `state_dict()`."""
    adapter = CDiffMRBaseline(resolution=32, num_steps=5)
    before_keys = set(adapter.state_dict())
    before_children = set(dict(adapter.named_children()))

    adapter.training_loss(torch.randn(1, 2, 32, 32))  # builds `_process()` lazily

    assert set(adapter.state_dict()) == before_keys, (
        "building the upstream GaussianDiffusion changed state_dict() -- it "
        "registered as a submodule and doubled the checkpoint"
    )
    assert set(dict(adapter.named_children())) == before_children

    fresh = CDiffMRBaseline(resolution=32, num_steps=5)
    fresh.load_state_dict(
        {k: v.clone() for k, v in adapter.state_dict().items()}, strict=True
    )


def test_shen2024_upstream_process_does_not_double_the_checkpoint() -> None:
    """Building the lazy `KspaceDiffusion` must not change `state_dict()`."""
    adapter = Shen2024Baseline(image_size=32, timesteps=5)
    before_keys = set(adapter.state_dict())
    before_children = set(dict(adapter.named_children()))

    adapter.training_loss(torch.randn(1, 2, 32, 32))  # builds `_process()` lazily

    assert set(adapter.state_dict()) == before_keys, (
        "building the upstream KspaceDiffusion changed state_dict() -- it "
        "registered as a submodule and doubled the checkpoint"
    )
    assert set(dict(adapter.named_children())) == before_children

    fresh = Shen2024Baseline(image_size=32, timesteps=5)
    fresh.load_state_dict(
        {k: v.clone() for k, v in adapter.state_dict().items()}, strict=True
    )


def test_fdb_declares_the_input_invariant_probe_skip() -> None:
    """FINDING 23: an untrained FDB UNet is exactly zero-init -- declare it, don't hide it.

    ``_coerce_probe_skip`` is the function `spectramr.infrastructure.validation.forward_probe`
    reads this attribute through; asserting against IT (rather than the raw class attribute)
    is what proves the declaration actually reaches the probe's opt-out, not merely that the
    attribute exists under a plausible name.
    """
    from spectramr.infrastructure.validation.forward_probe import _coerce_probe_skip

    assert _coerce_probe_skip(FDBBaseline.synthetic_forward_probe_skip) == {"input_invariant"}


@needs_fdb
def test_fdb_fresh_init_is_the_zero_output_the_skip_documents() -> None:
    """The fact the skip's comment claims, pinned: zero regardless of input.

    Upstream's ``UNetModel`` ends in ``zero_module(conv_nd(...))``
    (``external/baselines/fdb/utils/unet.py:436``), which is what makes the declared
    skip correct rather than a cover-up: a fresh instance really is input-invariant,
    by construction, before a single optimizer step.
    """
    adapter = FDBBaseline(image_size=32, bridge_steps=5).eval()
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        y1 = adapter(torch.randn(1, 2, 32, 32), t)
        y2 = adapter(torch.randn(1, 2, 32, 32) * 5, t)
    assert float(y1.std()) == 0.0
    assert float((y1 - y2).abs().mean()) == 0.0


@needs_fdb
def test_fdb_diffusion_bridge_was_never_an_nn_module() -> None:
    """FDB's own bridge is unaffected -- pins the fix's scope.

    `DiffusionBridge` (unlike `GaussianDiffusion` and `KspaceDiffusion`) is a
    plain object, not an `nn.Module`, so assigning it through
    `self._diffusion = diffusion` never registered a child in the first place.
    """
    adapter = FDBBaseline(image_size=32, bridge_steps=5)
    assert not isinstance(adapter._diffusion, torch.nn.Module)
    assert list(dict(adapter.named_children())) == ["upstream"]


def test_a_plain_module_assignment_would_have_registered_the_child() -> None:
    """PLANTED VIOLATION: proves the two tests above are not vacuous.

    A `torch.nn.Module` assigned through the ordinary `self.x = value` path --
    the shape the fix removes -- DOES change `state_dict()`. If it stopped, the
    assertions above would never go red on a regression.
    """

    class _Holder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _Outer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner: torch.nn.Module | None = None

        def build_unfixed(self) -> None:
            self.inner = _Holder()  # plain assignment -- registers as a child

    outer = _Outer()
    before = set(outer.state_dict())
    outer.build_unfixed()
    after = set(outer.state_dict())
    assert after != before, (
        "a plain `self.x = nn.Module()` no longer registers as a child on this "
        "torch version -- the regression this fix guards against can no "
        "longer take this shape"
    )


@needs_cdiffmr
def test_cdiffmr_upstream_module_loaded_via_importlib() -> None:
    """CDiffMR upstream is loaded via ``importlib.util`` — not via sys.path pollution.

    Regression: previously we considered adding the CDiffMR root to
    ``sys.path``, which would also expose ``models`` and ``utils`` as
    top-level packages and shadow this repo's modules of the same name.
    Importlib loads only the single ``network_cdiff_unet2.py`` file.
    """
    import importlib
    importlib.invalidate_caches()
    # The sentinel module name must be present once the adapter loaded.
    from spectramr.models.baselines.cdiffmr import _load_upstream_model_class
    _load_upstream_model_class()
    assert "_cdiffmr_upstream_network" in sys.modules
    # The upstream's "models" and "utils" packages must NOT have leaked.
    if "models" in sys.modules:
        mod_file = getattr(sys.modules["models"], "__file__", "")
        assert "external/baselines/cdiffmr" not in str(mod_file), (
            "CDiffMR's top-level `models` package has polluted sys.modules"
        )


@needs_fdb
def test_fdb_upstream_path_injection_is_idempotent() -> None:
    """Repeated calls to ``_ensure_upstream_on_sys_path`` don't multiply entries."""
    from spectramr.models.baselines.fdb import _FDB_DIR, _ensure_upstream_on_sys_path

    _ensure_upstream_on_sys_path()
    n_before = sys.path.count(str(_FDB_DIR))
    _ensure_upstream_on_sys_path()
    _ensure_upstream_on_sys_path()
    assert sys.path.count(str(_FDB_DIR)) == n_before


@pytest.mark.parametrize(
    "cls", [CDiffMRBaseline, Shen2024Baseline, FDBBaseline],
)
def test_provenance_dict_complete_for_each_adapter(cls: type) -> None:
    """``baseline_provenance`` emits a complete record for every adapter."""
    prov = baseline_provenance(cls)
    assert prov["repo_name"] == cls.REPO_NAME
    assert prov["paper_ref"] == cls.PAPER_REF
    assert prov["preferred_mask_type"] == cls.PREFERRED_MASK_TYPE
    assert prov["preferred_fft_norm"] == "ortho"


@needs_cdiffmr
def test_cdiffmr_constructor_reads_its_own_declared_knobs() -> None:
    """The constructor takes a sampler-step-count + channel knobs.

    Renamed from ``..._accepts_paradigm_kwargs``. It never tested that -- every
    argument below is a real parameter of the signature -- but the name read as
    an assertion that ``kspace_cold_diffusion``'s knob names are welcome here,
    which is the ``**kwargs`` swallowing that let `baseline_cdiffmr` declare nine
    keys and have one read. What the adapter does with an unrecognised name is
    pinned by ``test_unconsumed_model_kwargs_are_reported_in_provenance``.
    """
    adapter = CDiffMRBaseline(
        in_channels=2, out_channels=2, resolution=32, num_steps=25,
    )
    assert adapter.num_steps == 25


@needs_fdb
def test_fdb_constructor_accepts_bridge_kwargs() -> None:
    """The FDB constructor accepts bridge-schedule hyperparameters."""
    adapter = FDBBaseline(image_size=64, bridge_steps=200)
    assert adapter.bridge_steps == 200


@needs_fdb
def test_fdb_bridge_drift_is_no_longer_a_silent_absorb() -> None:
    """FINDING 22: `bridge_drift` named nothing upstream and is now DELETED.

    `DiffusionBridge.__init__` takes only `(steps, undersampling_rate,
    image_size, data_type)` -- its schedule comes from `undersampling_rate` and
    the `w.npy` calibration, so there was no upstream parameter to forward this
    to. It used to be a NAMED constructor argument, stored and never read
    (pitfall 15); a caller still spelling it now gets the honest outcome every
    other unrecognised `model_kwargs` key gets -- reported as dropped, not
    silently absorbed into an attribute nothing consulted.
    """
    adapter = FDBBaseline(image_size=64, bridge_steps=200, bridge_drift=0.5)
    assert not hasattr(adapter, "bridge_drift")
    assert "bridge_drift" in adapter.provenance()["unconsumed_model_kwargs"]


@needs_fdb
def test_fdb_sample_loop_runs_end_to_end_on_synthetic_input() -> None:
    """The full bridge sampling loop returns a tensor of the requested shape.

    Without trained weights the output is noise — but the plumbing must
    work so a campaign can later drop in checkpoints. The test uses tiny
    image_size + 3 bridge steps to keep wall-clock under a second.
    """
    adapter = FDBBaseline(image_size=32, bridge_steps=3)
    kspace = torch.randn(1, 2, 32, 32)
    mask = torch.ones(1, 2, 32, 32)
    out = adapter.sample(kspace, mask, batch_size=1)
    assert out.shape == (1, 2, 32, 32)


@needs_fdb
def test_fdb_validation_sample_raises_naming_the_missing_mask() -> None:
    """FINDING 21: FDB's reverse loop needs the loader's REAL mask -- not guessed.

    `DiffusionBridge.p_sample_loop_condition` derives its reverse-timestep mask
    ladder from the true undersampling pattern, which `(input_batch, target_batch)`
    alone does not carry -- this must say so, not silently drive the loop off a
    fabricated mask that would report a plausible but unverified number.
    """
    adapter = FDBBaseline(image_size=32, bridge_steps=3)
    x = torch.randn(1, 2, 32, 32)
    with pytest.raises(NotImplementedError, match="loader's real undersampling mask"):
        adapter.validation_sample(input_batch=x, target_batch=x)


@needs_cdiffmr
def test_cdiffmr_sample_raises_off_an_accelerator() -> None:
    """FINDING 21/upstream `.cuda()` hard-code: the reverse loop refuses on CPU, loudly.

    ``GaussianDiffusion.sample`` hard-codes ``.cuda()`` on its per-step timestep
    tensor, so this adapter's own ``sample()`` guards it first rather than let a bare
    unrelated CUDA error surface with no context.
    """
    adapter = CDiffMRBaseline(resolution=32)
    x = torch.randn(1, 2, 32, 32)
    with pytest.raises(RuntimeError, match=r"hard-codes `\.cuda\(\)`"):
        adapter.sample(x_start=x)


@needs_cdiffmr
@pytest.mark.skipif(not _cuda_actually_usable(), reason="no usable CUDA device on this box")
def test_cdiffmr_sample_runs_the_real_reverse_loop_on_an_accelerator() -> None:
    """On a working accelerator, `sample()` is wired for real, not deferred."""
    adapter = CDiffMRBaseline(resolution=32, num_steps=3).cuda()
    x = torch.randn(1, 2, 32, 32, device="cuda")
    out = adapter.sample(x_start=x)
    assert out.shape == x.shape


@needs_cdiffmr
def test_cdiffmr_validation_sample_delegates_to_sample() -> None:
    """`validation_sample` is `sample()` driven from the clean target -- same guard."""
    adapter = CDiffMRBaseline(resolution=32)
    x = torch.randn(1, 2, 32, 32)
    with pytest.raises(RuntimeError, match=r"hard-codes `\.cuda\(\)`"):
        adapter.validation_sample(input_batch=torch.zeros_like(x), target_batch=x)


@pytest.mark.parametrize(
    "name,expected_cls",
    [
        ("cdiffmr_baseline", CDiffMRBaseline),
        ("shen2024_baseline", Shen2024Baseline),
        ("fdb_baseline", FDBBaseline),
    ],
)
def test_baseline_name_resolves_to_real_adapter_not_stub(name, expected_cls) -> None:
    """Regression: each baseline name resolves to its real BaselineAdapter
    implementation, not the removed ``_BackboneAlias`` placeholder that used to
    live in ``_v6_1_registrations.py`` and double-registered the name (tripping
    the fail-loud collision guard during collection).
    """
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import get_model_class

    populate_model_registry()  # idempotent; must not raise a collision
    resolved = get_model_class(name)
    assert resolved is expected_cls
    assert issubclass(resolved, BaselineAdapter)


def test_repo_root_resolves_to_the_directory_holding_external_baselines() -> None:
    """The adapters' `_REPO_ROOT` must be the repo root, not `src/`.

    Regression for the `src -> src/spectramr` refactor (2026-05). These modules
    moved from `src/models/baselines/` to `src/spectramr/models/baselines/`, one
    level deeper, but kept `parents[3]` — which stopped being the repo root and
    became `src/`. The vendored upstreams then looked absent, and the adapters
    raised an error instructing the user to run a `git submodule add` they had
    already run, against a path (`src/external/baselines/...`) that has never
    been where the submodules live.

    Asserted against `pyproject.toml` rather than a parent count, so the next
    move of these files fails here with a clear reason instead of re-breaking
    the lookup.
    """
    from spectramr.models.baselines.cdiffmr import _CDIFFMR_DIR, _REPO_ROOT as CD_ROOT
    from spectramr.models.baselines.fdb import _FDB_DIR, _REPO_ROOT as FDB_ROOT

    for root in (CD_ROOT, FDB_ROOT):
        assert (root / "pyproject.toml").is_file(), (
            f"_REPO_ROOT={root} is not the repository root; "
            "the parents[] index is wrong for this file's depth"
        )

    for upstream in (_CDIFFMR_DIR, _FDB_DIR):
        assert upstream.parent.name == "baselines"
        assert upstream.parent.parent.name == "external"
        assert "src" not in upstream.parts, (
            f"{upstream} points inside the package; the vendored submodules live "
            "at <repo>/external/baselines/"
        )


@pytest.mark.parametrize("cls", [CDiffMRBaseline, Shen2024Baseline, FDBBaseline])
def test_unconsumed_model_kwargs_are_reported_in_provenance(cls: type) -> None:
    """A model_kwargs key that names no parameter must be visible, not vanish.

    Every adapter signature ends in ``**kwargs``, so an unrecognised YAML key is
    accepted and dropped in silence. That is how ``baseline_fdb`` declared eleven
    knobs of which one was read, and how ``timesteps`` sat beside a parameter
    spelled ``bridge_steps``, agreeing only because both defaulted to 1000.

    The adapters deliberately still ACCEPT them -- the factory injects
    framework-side kwargs no signature names, so refusing outright would refuse
    every arm -- but the drop is now recorded and lands in the run summary.

    Planted here rather than observed later (non-negotiable 15): this asserts the
    reporting fires on a name chosen to look plausible, which is the shape the
    real defect took.
    """
    # Real names from the two arms, neither a parameter of any adapter. `base_channels`
    # used to be in this set and left it on 2026-09-15: wiring Shen's `Unet` made it a
    # genuine parameter there, so planting it would assert that a CONSUMED knob is
    # reported as dropped -- the opposite of the rule.
    planted = {"condition_dim": 64, "bridge_alpha_min": 0.1}
    if cls is CDiffMRBaseline:
        adapter = cls(resolution=32, **planted)
    elif cls is FDBBaseline:
        adapter = cls(image_size=64, bridge_steps=10, **planted)
    else:
        adapter = cls(image_size=64, timesteps=8, **planted)

    reported = adapter.provenance()["unconsumed_model_kwargs"]
    for name in planted:
        assert name in reported, f"{cls.__name__} dropped {name!r} and did not report it"
    assert "bridge_steps" not in reported, "a CONSUMED knob must not be reported as dropped"


def test_provenance_reports_empty_string_when_nothing_was_dropped() -> None:
    """Consumed-everything and never-asked must not render alike.

    An absent key would be indistinguishable from an adapter that predates the
    recorder, so the field is always present and empty when there is nothing to
    say (the same reasoning as the optional-import rule: absent is a state to
    report, never one to infer).
    """
    adapter = Shen2024Baseline(in_channels=2, out_channels=2, image_size=64, timesteps=8)
    assert adapter.provenance()["unconsumed_model_kwargs"] == ""


# ---------------------------------------------------------------------------
# The timestep must survive the strategies' calling convention
# ---------------------------------------------------------------------------
#
# Both `DiffusionTrainingStrategy` and `GraphColdDiffusionStrategy` decide how to
# pass the timestep with `_callable_accepts_kwarg(forward, "timesteps")`, which
# returns True for ANY `**kwargs` signature -- it answers "will this call raise?",
# not "will this be consumed". Under the adapters' original parameter name `t`
# the value landed in `**kwargs` and was discarded, so every step trained at
# t=0 while the call looked correct from both sides (#2086).


class _SpyUpstream(torch.nn.Module):
    """Records the timestep the upstream network is handed."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[int | None] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        self.seen.append(None if t is None else int(torch.as_tensor(t).flatten()[0]))
        return x


def _drive(adapter: object, *call_args: object, **call_kwargs: object) -> int | None:
    x = torch.zeros(1, 2, 8, 8)
    with torch.no_grad():
        adapter(x, *call_args, **call_kwargs)  # type: ignore[operator]
    return adapter.upstream.seen[-1]  # type: ignore[attr-defined]


@pytest.mark.parametrize("cls", [CDiffMRBaseline, FDBBaseline])
@pytest.mark.parametrize("call", ["timesteps", "t", "positional"])
def test_the_timestep_reaches_the_upstream_network(cls: type, call: str) -> None:
    adapter = cls.__new__(cls)
    torch.nn.Module.__init__(adapter)
    adapter.upstream = _SpyUpstream()
    # `_drive` sends an 8x8 input; CDiffMRBaseline.forward now guards
    # `x.shape[2:] == (self.resolution, self.resolution)` before calling
    # upstream (the bare-AssertionError fix), so this bypassed-`__init__`
    # fixture must set it explicitly rather than rely on a default.
    adapter.resolution = 8
    t900 = torch.full((1,), 900, dtype=torch.long)
    seen = _drive(adapter, timesteps=t900) if call == "timesteps" else (
        _drive(adapter, t=t900) if call == "t" else _drive(adapter, t900)
    )
    assert seen == 900, f"{cls.__name__} dropped the timestep when passed as {call!r}"


def test_cdiffmr_resolution_mismatch_raises_with_both_values_named() -> None:
    """A patch_size/resolution mismatch must name both sides, not bare-assert.

    Upstream's own guard is ``assert x.shape[2] == x.shape[3] == self.resolution``
    with an EMPTY message (`network_cdiff_unet2.py:306`); this adapter's own
    ``self.resolution`` is stored and was never checked before calling in, so the
    failure surfaced with no context at all.
    """
    adapter = CDiffMRBaseline.__new__(CDiffMRBaseline)
    torch.nn.Module.__init__(adapter)
    adapter.resolution = 32
    adapter.out_channels = 2
    adapter.upstream = _SpyUpstream()
    with pytest.raises(ValueError, match=r"model\.model_kwargs\.resolution"):
        adapter(torch.zeros(1, 2, 8, 8))


def test_the_old_parameter_name_is_what_dropped_it() -> None:
    """The violation this gate exists for, planted.

    An adapter spelling its timestep `t` alongside `**kwargs` swallows the
    strategies' `timesteps=` silently -- no raise, no warning, upstream sees 0.
    """

    class _OldStyle(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.upstream = _SpyUpstream()

        def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor | None = None,
            **kwargs: object,
        ) -> torch.Tensor:
            if t is None:
                t = torch.zeros(x.shape[0], dtype=torch.long)
            return self.upstream(x, t)

    from spectramr.infrastructure.training.strategies.mixins.utils import (
        _callable_accepts_kwarg,
    )

    old = _OldStyle()
    # The predicate says yes, so the strategies pass `timesteps=` ...
    assert _callable_accepts_kwarg(old.forward, "timesteps") is True
    # ... and it is swallowed: upstream sees 0, not 900.
    assert _drive(old, timesteps=torch.full((1,), 900, dtype=torch.long)) == 0
